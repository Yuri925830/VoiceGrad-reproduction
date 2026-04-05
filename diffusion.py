import torch
import torch.nn as nn
import math


class VoiceGradDiffusion(nn.Module):
    def __init__(self, n_levels=20, offset=0.008):
        super().__init__()
        self.n_levels = n_levels
        self.offset = offset

        # =========================================================
        # 1. Cosine Noise Schedule (严格按论文 V-E 公式 21)
        #
        # alpha_bar_l = f(l) / f(0)
        # f(l) = cos(((l / L) + eta) / (1 + eta) * pi / 2)^2
        #
        # beta_l = 1 - alpha_bar_l / alpha_bar_{l-1}
        # beta_l clipped to <= 0.999
        # =========================================================
        steps = torch.arange(n_levels + 1, dtype=torch.float64) / n_levels

        f = torch.cos(
            ((steps + offset) / (1.0 + offset)) * (math.pi / 2.0)
        ) ** 2

        alpha_bar = f / f[0]  # shape: [L+1], alpha_bar[0] = 1

        betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])  # shape: [L]
        betas = torch.clamp(betas, min=0.0, max=0.999)

        betas = betas.float()
        alphas = (1.0 - betas).float()

        # 注意：
        # 这里使用 clipped beta 重新累乘得到 alpha_bar，
        # 这样训练和采样严格自洽。
        alphas_cumprod = torch.cumprod(alphas, dim=0).float()

        # =========================================================
        # 2. 预计算常用系数
        # =========================================================
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        self.register_buffer("sqrt_alphas", torch.sqrt(alphas))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer("recip_sqrt_alphas", 1.0 / torch.sqrt(alphas))

        # Algorithm 4 里的系数:
        # (1 - alpha_l) / sqrt(1 - alpha_bar_l)
        self.register_buffer(
            "remove_noise_coeff",
            betas / torch.sqrt(1.0 - alphas_cumprod)
        )

        # 论文中设 nu_l^2 = beta_l，所以 nu_l = sqrt(beta_l)
        self.register_buffer("sigma", torch.sqrt(betas))

    def get_index(self, tensor, t, shape):
        """
        从 shape [L] 的时间表里，取出 batch 中每个样本在时刻 t 的值，
        然后 reshape 成可广播形状。
        """
        batch_size = t.shape[0]
        out = tensor.gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        """
        前向扩散（训练时用）
        论文公式:
            x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_bar_t = self.get_index(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alpha_bar_t = self.get_index(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise

    @torch.no_grad()
    def sample(self, model, x_source, speaker_id, bnf, start_level=11):
        """
        严格按论文 Algorithm 4 的 DPM-based VoiceGrad 采样

        Args:
            model: 训练好的 VoiceGrad 模型
            x_source: 源语音 mel（已归一化） [B, 80, T]
            speaker_id: 目标说话人 ID [B]
            bnf: BNF 特征 [B, 144, T]
            start_level: 论文中的 L'，默认 11

        Returns:
            x: 转换后的 mel 特征
        """
        if start_level < 1 or start_level > self.n_levels:
            raise ValueError(
                f"start_level must be in [1, {self.n_levels}], but got {start_level}"
            )

        # 论文思路：直接从 source mel 开始反向扩散
        x = x_source.clone()
        batch_size = x.shape[0]
        device = x.device

        # 论文写法是 l = L' ... 1
        # 这里显式保留论文的 1-based 语义，再映射到 0-based index
        for l in range(start_level, 0, -1):
            t = torch.full((batch_size,), l - 1, device=device, dtype=torch.long)

            # epsilon_theta(x, l, k)
            predicted_noise = model(x, t, speaker_id, bnf)

            recip_sqrt_alpha = self.get_index(self.recip_sqrt_alphas, t, x.shape)
            noise_coeff = self.get_index(self.remove_noise_coeff, t, x.shape)
            sigma = self.get_index(self.sigma, t, x.shape)

            # Algorithm 4:
            # x <- 1/sqrt(alpha_l) * (x - (1-alpha_l)/sqrt(1-alpha_bar_l) * eps_theta) + nu_l * z
            mean = recip_sqrt_alpha * (x - noise_coeff * predicted_noise)

            # 严格按论文：每一步都 draw z ~ N(0, I)
            z = torch.randn_like(x)

            x = mean + sigma * z

        return x


if __name__ == "__main__":
    # =========================
    # 最小自测
    # =========================
    diffusion = VoiceGradDiffusion(n_levels=20, offset=0.008)

    batch_size = 2
    x0 = torch.randn(batch_size, 80, 128)
    t = torch.randint(0, 20, (batch_size,))
    xt = diffusion.q_sample(x0, t)

    print("x0 shape:", x0.shape)
    print("xt shape:", xt.shape)

    # 检查 schedule 合法性
    print("betas shape:", diffusion.betas.shape)
    print("alphas_cumprod shape:", diffusion.alphas_cumprod.shape)
    print("beta min/max:", diffusion.betas.min().item(), diffusion.betas.max().item())