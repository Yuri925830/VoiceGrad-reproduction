import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import re


class VoiceGradDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split='train',
        segment_length=128,
        sample_rate=16000,
        hop_size=256,
        bnf_frame_shift_ms=10.0
    ):
        self.root_dir = root_dir
        self.segment_length = segment_length
        self.mel_dir = os.path.join(root_dir, 'mel')
        self.bnf_dir = os.path.join(root_dir, 'bnf')
        self.split = split

        # ===== 时间轴参数 =====
        # mel: 16k / hop_size=256 -> 62.5 fps
        # bnf: frame_shift=10ms -> 100 fps
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.mel_fps = sample_rate / hop_size
        self.bnf_frame_shift_ms = bnf_frame_shift_ms
        self.bnf_fps = 1000.0 / bnf_frame_shift_ms

        # 闭集 (训练/目标) 说话人 K=4
        self.train_speakers = ['clb', 'bdl', 'slt', 'rms']
        # 开集 (仅测试源) 说话人
        self.openset_speakers = ['jmk', 'ksp', 'lnh']

        # 建立固定的 ID 映射
        self.spk2id = {spk: i for i, spk in enumerate(self.train_speakers)}

        # 加载统计量
        stats_dir = os.path.join(root_dir, 'stats')
        if not os.path.exists(os.path.join(stats_dir, 'mel_mean.npy')):
            print("【Warning】未找到统计文件，使用临时均值...")
            self.mel_mean = torch.zeros(80, 1)
            self.mel_std = torch.ones(80, 1)
        else:
            self.mel_mean = torch.from_numpy(
                np.load(os.path.join(stats_dir, 'mel_mean.npy'))
            ).float().view(-1, 1)
            self.mel_std = torch.from_numpy(
                np.load(os.path.join(stats_dir, 'mel_std.npy'))
            ).float().view(-1, 1)

        self.file_list = []

        all_speakers = sorted([
            d for d in os.listdir(self.mel_dir)
            if os.path.isdir(os.path.join(self.mel_dir, d))
        ])

        debug_counter = 0

        for spk in all_speakers:
            spk_mel_dir = os.path.join(self.mel_dir, spk)
            spk_bnf_dir = os.path.join(self.bnf_dir, spk)

            files = [f for f in os.listdir(spk_mel_dir) if f.endswith('.npy')]
            files.sort()

            for f in files:
                try:
                    nums = re.findall(r'\d+', f)
                    local_idx = int(nums[-1])

                    # CMU Arctic Set A = 593 句
                    if 'arctic_b' in f or '_b' in f:
                        global_idx = local_idx + 593
                    else:
                        global_idx = local_idx

                except Exception:
                    print(f"Warning: Skip {f}, parse error.")
                    continue

                is_valid = self._is_file_in_split(spk, global_idx, split)

                if split == 'val' and is_valid and debug_counter < 3:
                    print(f"[Val Debug] Keep {spk} {f} -> Global Idx {global_idx}")
                    debug_counter += 1

                if is_valid:
                    bnf_name = f.replace('.npy', '.ling_feat.npy')

                    # 容错：如果找不到 BNF，尝试同名
                    if not os.path.exists(os.path.join(spk_bnf_dir, bnf_name)):
                        bnf_name = f

                    bnf_path = os.path.join(spk_bnf_dir, bnf_name)
                    if os.path.exists(bnf_path):
                        spk_id = self.spk2id[spk] if spk in self.spk2id else -1
                        self.file_list.append({
                            'mel_path': os.path.join(spk_mel_dir, f),
                            'bnf_path': bnf_path,
                            'spk_id': spk_id,
                            'spk_name': spk,
                            'global_idx': global_idx
                        })

        print(f"Dataset split: {split} | Samples: {len(self.file_list)}")
        print(
            f"[Time Axis] mel_fps={self.mel_fps:.4f}, "
            f"bnf_fps={self.bnf_fps:.4f}, "
            f"expected_ratio={self.bnf_fps / self.mel_fps:.4f}"
        )

        if split == 'train' and len(self.file_list) != 1000:
            print(f"【注意】训练集数量为 {len(self.file_list)}，预期为 1000。请检查文件完整性。")

    def _is_file_in_split(self, spk, idx, split):
        # idx 是 1-1132 的全局索引

        # 1. TEST SET (1101 - 1132)
        if split == 'test':
            if 1101 <= idx <= 1132:
                if spk in self.train_speakers or spk in self.openset_speakers:
                    return True
            return False

        # 2. VALIDATION SET (1001 - 1100)
        if split == 'val':
            if 1001 <= idx <= 1100:
                if spk in self.train_speakers:
                    return True
            return False

        # 3. TRAINING SET (1 - 1000) - 非平行严格划分
        if split == 'train':
            if not (1 <= idx <= 1000):
                return False

            if spk == 'clb' and (1 <= idx <= 250):
                return True
            if spk == 'bdl' and (251 <= idx <= 500):
                return True
            if spk == 'slt' and (501 <= idx <= 750):
                return True
            if spk == 'rms' and (751 <= idx <= 1000):
                return True

            return False

        return False

    def __len__(self):
        return len(self.file_list)

    def _ensure_mel_shape(self, mel):
        # 目标形状: [80, T]
        if mel.ndim != 2:
            raise ValueError(f"mel ndim should be 2, but got {mel.shape}")
        if mel.shape[0] == 80:
            return mel
        if mel.shape[1] == 80:
            return mel.T
        raise ValueError(f"Invalid mel shape: {mel.shape}")

    def _ensure_bnf_shape(self, bnf):
        # 目标形状: [144, T]
        if bnf.ndim != 2:
            raise ValueError(f"bnf ndim should be 2, but got {bnf.shape}")
        if bnf.shape[0] == 144:
            return bnf
        if bnf.shape[1] == 144:
            return bnf.T
        raise ValueError(f"Invalid bnf shape: {bnf.shape}")

    def _resample_bnf_to_mel_length(self, bnf, target_len):
        """
        将 BNF 从 [144, T_bnf] 按时间轴重采样到 [144, target_len]
        这里不是“硬裁剪”，而是解决 1.6 倍时间轴错位。
        """
        current_len = bnf.shape[1]

        if current_len == target_len:
            return bnf

        if current_len <= 1:
            # 极端容错，直接重复到目标长度
            return np.repeat(bnf, target_len, axis=1)

        bnf_tensor = torch.from_numpy(bnf).float().unsqueeze(0)  # [1, 144, T_bnf]

        # 1D 时间轴线性插值
        bnf_tensor = F.interpolate(
            bnf_tensor,
            size=target_len,
            mode='linear',
            align_corners=False
        )

        return bnf_tensor.squeeze(0).cpu().numpy()

    def __getitem__(self, idx):
        item = self.file_list[idx]

        try:
            mel = np.load(item['mel_path'])
            bnf = np.load(item['bnf_path'])
        except Exception:
            return self.__getitem__(random.randint(0, len(self.file_list) - 1))

        try:
            mel = self._ensure_mel_shape(mel)
            bnf = self._ensure_bnf_shape(bnf)
        except Exception as e:
            print(f"[Shape Error] {item['mel_path']} / {item['bnf_path']} -> {e}")
            return self.__getitem__(random.randint(0, len(self.file_list) - 1))

        # ===== 关键修复 =====
        # 原版做法是 min_len 硬裁剪，这会把 1.6 倍错位藏起来。
        # 这里改成：先把 BNF 按时间轴重采样到 mel 的长度。
        mel_len = mel.shape[1]
        bnf = self._resample_bnf_to_mel_length(bnf, mel_len)

        # 现在 mel 和 bnf 的时间长度严格一致
        assert mel.shape[1] == bnf.shape[1], \
            f"Length mismatch after resample: mel={mel.shape}, bnf={bnf.shape}"

        total_len = mel.shape[1]

        if self.segment_length is not None:
            if total_len > self.segment_length:
                start = random.randint(0, total_len - self.segment_length)
                end = start + self.segment_length
                mel = mel[:, start:end]
                bnf = bnf[:, start:end]
            else:
                pad_len = self.segment_length - total_len
                mel = np.pad(mel, ((0, 0), (0, pad_len)), mode='constant')
                bnf = np.pad(bnf, ((0, 0), (0, pad_len)), mode='constant')

        mel_tensor = torch.from_numpy(mel).float()
        bnf_tensor = torch.from_numpy(bnf).float()

        mel_normalized = (mel_tensor - self.mel_mean) / self.mel_std

        return {
            'mel': mel_normalized,
            'bnf': bnf_tensor,
            'spk_id': torch.tensor(item['spk_id']).long(),
            'spk_name': item['spk_name']
        }


def get_dataloader(root_dir, split, batch_size, num_workers=4):
    seg_len = 128 if split == 'train' else None
    dataset = VoiceGradDataset(
        root_dir=root_dir,
        split=split,
        segment_length=seg_len,
        sample_rate=16000,
        hop_size=256,
        bnf_frame_shift_ms=10.0
    )
    bs = batch_size if split == 'train' else 1
    shuffle = (split == 'train')
    return DataLoader(dataset, batch_size=bs, shuffle=shuffle, num_workers=num_workers)