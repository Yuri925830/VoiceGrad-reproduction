import torch
import torch.nn as nn
import torch.nn.functional as F


class VoiceGrad(nn.Module):
    def __init__(
        self,
        n_mels=80,
        n_bnf=144,
        n_channels=512,
        n_spk=18,
        n_levels=20,
        cond_dim=128,
        bnf_out_dim=32
    ):
        """
        VoiceGrad Score Approximator Model.

        关键修复：
        1. dataset.py 已经把输入 BNF 对齐到 mel 的原始时间长度 T。
        2. 这里每一层再根据“该层输入长度”选择合适的 BNF stride。
        3. 不再使用 F.interpolate 对 BNF 特征做补锅式强行拉齐。
        """
        super().__init__()

        self.n_channels = n_channels

        # 条件嵌入层
        self.noise_emb = nn.Embedding(n_levels + 1, cond_dim)
        self.spk_emb = nn.Embedding(n_spk, cond_dim)

        # =========================
        # BNF stride 设计原则（按“该层输入长度”来定）
        #
        # layer1 输入长度: T    -> bnf_stride = 1
        # layer2 输入长度: T    -> bnf_stride = 1
        # layer3 输入长度: T/2  -> bnf_stride = 2
        # layer4 输入长度: T/2  -> bnf_stride = 2
        # layer5 输入长度: T/4  -> bnf_stride = 4
        # layer6 输入长度: T/4  -> bnf_stride = 4
        # layer7 输入长度: T/4  -> bnf_stride = 4
        # layer8 输入长度: T/4  -> bnf_stride = 4
        # layer9 输入长度: T/2  -> bnf_stride = 2
        # layer10 输入长度: T/2 -> bnf_stride = 2
        # layer11 输入长度: T   -> bnf_stride = 1
        #
        # 这才符合论文里“stride r 选到和该层输入长度兼容”的意思。
        # =========================

        # Encoder
        self.layer1 = VoiceGradBlock(
            n_mels, n_channels, k=9, s=1,
            cond_dim=cond_dim, bnf_dim=n_bnf, bnf_out=bnf_out_dim,
            bnf_stride=1
        )
        self.layer2 = VoiceGradBlock(
            n_channels, n_channels, k=8, s=2,
            cond_dim=cond_dim, bnf_dim=n_bnf, bnf_out=bnf_out_dim,
            bnf_stride=1
        )  # input len = T
        self.layer3 = VoiceGradBlock(
            n_channels, n_channels, k=9, s=1,
            cond_dim=cond_dim, bnf_dim=n_bnf, bnf_out=bnf_out_dim,
            bnf_stride=2
        )  # input len = T/2
        self.layer4 = VoiceGradBlock(
            n_channels, n_channels, k=8, s=2,
            cond_dim=cond_dim, bnf_dim=n_bnf, bnf_out=bnf_out_dim,
            bnf_stride=2
        )  # input len = T/2
        self.layer5 = VoiceGradBlock(
            n_channels, n_channels, k=5, s=1,
            cond_dim=cond_dim, bnf_dim=n_bnf, bnf_out=bnf_out_dim,
            bnf_stride=4
        )  # input len = T/4
        self.layer6 = VoiceGradBlock(
            n_channels, n_channels, k=5, s=1,
            cond_dim=cond_dim, bnf_dim=n_bnf, bnf_out=bnf_out_dim,
            bnf_stride=4
        )  # input len = T/4

        # Decoder
        self.layer7 = VoiceGradBlock(
            n_channels, n_channels, k=5, s=1,
            cond_dim=cond_dim, bnf_dim=n_bnf, bnf_out=bnf_out_dim,
            bnf_stride=4
        )  # input len = T/4
        self.layer8 = VoiceGradBlock(
            n_channels, n_channels, k=8, s=2,
            cond_dim=cond_dim, bnf_dim=n_bnf, bnf_out=bnf_out_dim,
            bnf_stride=4, transpose=True
        )  # input len = T/4
        self.layer9 = VoiceGradBlock(
            n_channels, n_channels, k=9, s=1,
            cond_dim=cond_dim, bnf_dim=n_bnf, bnf_out=bnf_out_dim,
            bnf_stride=2
        )  # input len = T/2
        self.layer10 = VoiceGradBlock(
            n_channels, n_channels, k=8, s=2,
            cond_dim=cond_dim, bnf_dim=n_bnf, bnf_out=bnf_out_dim,
            bnf_stride=2, transpose=True
        )  # input len = T/2
        self.layer11 = VoiceGradBlock(
            n_channels, n_channels, k=9, s=1,
            cond_dim=cond_dim, bnf_dim=n_bnf, bnf_out=bnf_out_dim,
            bnf_stride=1
        )  # input len = T

        # 输出层
        self.final_conv = nn.utils.weight_norm(
            nn.Conv1d(n_channels, n_mels, kernel_size=9, padding=4)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.xavier_normal_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _match_shape(self, x, target):
        """
        用于 skip connection 的长度对齐。
        这里只做最小必要的 pad/crop，不做插值。
        """
        if x.shape[-1] < target.shape[-1]:
            diff = target.shape[-1] - x.shape[-1]
            x = F.pad(x, (0, diff))
        elif x.shape[-1] > target.shape[-1]:
            x = x[..., :target.shape[-1]]
        return x

    def forward(self, x, noise_level, speaker_id, bnf=None):
        if bnf is None:
            raise ValueError("BNF-conditioned VoiceGrad requires bnf input, but got None.")

        # noise embedding / speaker embedding
        n_emb = self.noise_emb(noise_level).unsqueeze(-1)   # [B, cond_dim, 1]
        s_emb = self.spk_emb(speaker_id).unsqueeze(-1)      # [B, cond_dim, 1]
        cond = torch.cat([n_emb, s_emb], dim=1)             # [B, 2*cond_dim, 1]

        # -------- Encoder --------
        out1 = self.layer1(x, cond, bnf)    # [B, C, T]
        out2 = self.layer2(out1, cond, bnf) # [B, C, T/2]
        out3 = self.layer3(out2, cond, bnf) # [B, C, T/2]
        out4 = self.layer4(out3, cond, bnf) # [B, C, T/4]
        out5 = self.layer5(out4, cond, bnf) # [B, C, T/4]
        out6 = self.layer6(out5, cond, bnf) # [B, C, T/4]

        # -------- Decoder --------
        out7 = self.layer7(out6, cond, bnf)
        out7 = self._match_shape(out7, out5) + out5

        out8 = self.layer8(out7, cond, bnf)
        out8 = self._match_shape(out8, out3) + out3

        out9 = self.layer9(out8, cond, bnf)
        out9 = self._match_shape(out9, out2) + out2

        out10 = self.layer10(out9, cond, bnf)
        out10 = self._match_shape(out10, out1) + out1

        out11 = self.layer11(out10, cond, bnf)

        out = self.final_conv(out11)
        return out


class VoiceGradBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        k,
        s,
        cond_dim,
        bnf_dim,
        bnf_out,
        bnf_stride=1,
        transpose=False
    ):
        super().__init__()
        self.transpose = transpose
        self.bnf_stride = bnf_stride

        # 关键修复：
        # 原版这里用和主卷积同样的 kernel，再加错误的 stride_factor，
        # 然后靠 F.interpolate 补锅。
        #
        # 这里改成一个“专门给 BNF 用的 strided conv”，只负责把时间分辨率压到当前层输入长度附近。
        # kernel_size=1 的好处是：
        # - 不会无缘无故因为偶数卷积核在 stride=1 时少一帧
        # - 对论文要求“32 通道 + stride r”是兼容的
        self.bnf_proj = nn.utils.weight_norm(
            nn.Conv1d(
                bnf_dim,
                bnf_out,
                kernel_size=1,
                stride=bnf_stride,
                padding=0
            )
        )

        total_in_ch = in_ch + (cond_dim * 2) + bnf_out
        glu_out_ch = out_ch * 2

        if transpose:
            # Deconv / Upsample
            padding = (k - s) // 2
            self.conv = nn.utils.weight_norm(
                nn.ConvTranspose1d(
                    total_in_ch,
                    glu_out_ch,
                    kernel_size=k,
                    stride=s,
                    padding=padding
                )
            )
        else:
            # Conv / Keep or Downsample
            padding = (k - 1) // 2
            self.conv = nn.utils.weight_norm(
                nn.Conv1d(
                    total_in_ch,
                    glu_out_ch,
                    kernel_size=k,
                    stride=s,
                    padding=padding
                )
            )

    def _match_time_length(self, x, target_len):
        """
        只做最小必要的 pad/crop。
        不再使用 interpolate 做补锅式时间拉伸。
        """
        if x.shape[-1] < target_len:
            diff = target_len - x.shape[-1]
            x = F.pad(x, (0, diff))
        elif x.shape[-1] > target_len:
            x = x[..., :target_len]
        return x

    def forward(self, x, cond, bnf):
        if bnf is None:
            raise ValueError("VoiceGradBlock requires bnf input, but got None.")

        T = x.shape[-1]

        # 把 noise/speaker embedding 在时间轴上重复到当前层输入长度
        cond_expanded = cond.expand(-1, -1, T)

        # BNF -> 当前层兼容长度
        bnf_feat = self.bnf_proj(bnf)

        # 正常情况下，长度应该已经基本兼容；
        # 奇数长度时可能只差 1 帧，这里只做最小 pad/crop
        if abs(bnf_feat.shape[-1] - T) > 1:
            raise RuntimeError(
                f"BNF length mismatch too large before match: "
                f"bnf_feat={bnf_feat.shape[-1]}, target={T}, bnf_stride={self.bnf_stride}"
            )

        bnf_feat = self._match_time_length(bnf_feat, T)

        net_in = torch.cat([x, cond_expanded, bnf_feat], dim=1)
        out = self.conv(net_in)
        out = F.glu(out, dim=1)
        return out


if __name__ == '__main__':
    # 简单自测：奇数长度输入
    batch_size = 2
    n_mels = 80
    time_steps = 161
    n_bnf = 144

    x = torch.randn(batch_size, n_mels, time_steps)
    # 关键：dataset 修复后，输入给模型的 bnf 时间长度应该和 mel 原始长度一致
    bnf = torch.randn(batch_size, n_bnf, time_steps)

    noise_idx = torch.randint(0, 20, (batch_size,))
    spk_idx = torch.randint(0, 4, (batch_size,))

    model = VoiceGrad(n_mels=80, n_bnf=144, n_channels=512, n_spk=4)
    output = model(x, noise_idx, spk_idx, bnf=bnf)

    print(f"Input: {x.shape}, BNF: {bnf.shape}, Output: {output.shape}")
    assert x.shape == output.shape
    print("Verification Passed: model handles odd lengths and aligned BNF correctly.")