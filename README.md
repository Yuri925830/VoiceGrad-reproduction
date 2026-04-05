# VoiceGrad (DPM + BNF) Reproduction

This repository contains my reproduction of the **DPM + BNF** version of **VoiceGrad**, a **non-parallel any-to-many voice conversion** model proposed in the original paper.  
The model is based on **Diffusion Probabilistic Models (DPM)**, **score matching**, and **annealed Langevin dynamics**, and uses **144-dimensional Bottleneck Features (BNF)** as linguistic conditioning.

After completing the full data preprocessing, training, inference, and evaluation pipeline, the reproduced model achieved **paper-level performance** under the closed-set CMU ARCTIC evaluation protocol.

## Overview

VoiceGrad performs voice conversion by learning a score approximator over speech features and iteratively denoising them toward the target speaker distribution.  
In this reproduction, the final system follows the DPM setting in the paper:

- **Input condition:** 144-dim BNF
- **Acoustic feature:** 80-dim log-mel spectrogram
- **Backbone:** U-Net-like 1D fully convolutional network
- **Speaker setting:** closed-set 4-speaker conversion
- **Vocoder:** self-trained **16 kHz HiFi-GAN**
- **Diffusion setting:** cosine noise schedule with `n_levels=20`

A key point of this reproduction is that the mel definition is made **fully consistent with the HiFi-GAN vocoder**, which is critical for obtaining stable and paper-level results.

## Dataset and Protocol

The reproduction follows the **CMU ARCTIC closed-set protocol** used in the VoiceGrad paper.

### Closed-set target speakers
- `clb`
- `bdl`
- `slt`
- `rms`

### Data split
Each speaker has 1132 utterances in total.

- **Train:** global indices `1-1000`
- **Validation:** `1001-1100`
- **Test:** `1101-1132`

For closed-set conversion evaluation, there are:

- **4 target speakers**
- **12 directed conversion pairs** (`4 × 3`)
- **32 test utterances per pair**
- **384 converted utterances in total**

The training setup is **non-parallel**, consistent with the paper.

## Preprocessing Pipeline

The full preprocessing pipeline is:

**raw wav -> 80-dim mel extraction -> mel mean/std statistics -> 144-dim BNF extraction -> BNF/mel temporal alignment -> VoiceGrad training -> mel denoising inference -> HiFi-GAN waveform synthesis**

### 1. Mel spectrograms
Mel spectrograms are extracted using the **official HiFi-GAN mel definition**, not an independently reimplemented one.  
This ensures that the acoustic space learned by VoiceGrad is exactly aligned with the vocoder input space.

Mel configuration:

- `sampling_rate = 16000`
- `num_mels = 80`
- `n_fft = 1024`
- `win_size = 1024`
- `hop_size = 256`
- `fmin = 0`
- `fmax = 8000`

### 2. Mel normalization
Global mel statistics are computed offline and saved as:

- `mel_mean.npy`
- `mel_std.npy`

During training, each mel spectrogram is normalized channel-wise.

### 3. Bottleneck Features (BNF)
BNF features are extracted from a pretrained **CTC-Attention Conformer PPG/ASR model**.  
The final BNF dimension used by the model is **144**.

### 4. Temporal alignment
Because mel and BNF are extracted at different frame rates, BNF is **linearly resampled** to match the mel length before training and inference.  
This temporal alignment is one of the important engineering fixes in this reproduction.

## Model

The final VoiceGrad model takes:

- noisy mel feature `x_t`
- diffusion level `t`
- target `speaker_id`
- source `BNF`

Main components:

- noise embedding
- speaker embedding
- BNF projection
- U-Net-like 1D convolutional backbone
- GLU-based VoiceGrad blocks

The model predicts the diffusion noise `epsilon_theta` with output shape `[B, 80, T]`.

## Diffusion Setting

The diffusion process is implemented with:

- `n_levels = 20`
- `offset = 0.008`
- **cosine noise schedule**

During inference, conversion does **not** start from pure Gaussian noise.  
Instead, following the DPM setting in the paper, the model starts from the **source mel perturbed at an intermediate diffusion level**:

- `start_level = 11`

This helps preserve source linguistic content and prosody while gradually shifting the acoustic characteristics toward the target speaker.

## Training Configuration

The final reproduced setting is:

- `epochs = 4000`
- `batch_size = 16`
- `learning_rate = 1e-4`
- `num_workers = 4`
- `seed = 1234`
- `grad_clip_max_norm = 1.0`
- `val_every = 20`
- `plot_every = 50`
- `save_every = 500`

Model setting:

- `n_mels = 80`
- `n_bnf = 144`
- `n_channels = 512`
- `n_spk = 4`

The training objective is the standard **L1 noise prediction loss** used in diffusion training.

## Inference and Vocoder

The final inference chain is:

**source mel + source BNF + target speaker_id -> reverse diffusion from start_level=11 -> converted mel -> mel denormalization -> HiFi-GAN -> waveform**

A self-trained **16 kHz HiFi-GAN** is used as the vocoder.

In this project, HiFi-GAN plays two roles:

1. waveform synthesis from converted mel
2. definition of the mel feature space itself

This consistency between the diffusion model and the vocoder is one of the main reasons the reproduction reaches the paper-reported level.

## Evaluation

The model is evaluated under the closed-set protocol using:

- **CER** ↓
- **MCD** ↓
- **LFC** ↑
- **pMOS** ↑

Among these metrics, **CER**, **MCD**, and **LFC** are treated as the main indicators.  
The pMOS result in this project is based on a DNSMOS-style pipeline and is used only as a relative reference.

## Main Results

### Checkpoint comparison

| Checkpoint | CER (%) ↓ | MCD (dB) ↓ | LFC ↑ | pMOS ↑ |
|-----------|-----------:|-----------:|------:|------:|
| ckpt_500  | 1.916 | 6.584 | 0.435 | 3.128 |
| ckpt_1000 | 1.955 | 6.107 | 0.468 | 3.201 |
| ckpt_1500 | 2.208 | 6.062 | 0.444 | 3.198 |
| ckpt_2000 | 2.209 | 6.048 | 0.472 | 3.196 |
| ckpt_2500 | 2.245 | 6.031 | 0.432 | 3.197 |
| ckpt_3000 | 2.211 | 6.008 | 0.416 | 3.195 |
| ckpt_3500 | 2.286 | 6.019 | 0.430 | 3.190 |
| ckpt_4000 | 2.252 | 6.016 | 0.429 | 3.179 |

### Recommended checkpoint
- **Best overall:** `ckpt_1000`
- **Lowest MCD only:** `ckpt_3000`

The automatically saved validation-loss-based `best_model.pt` was **not** the best model for final voice conversion quality.  
This is because validation loss monitors the diffusion noise prediction objective, which is only a proxy target and does not perfectly match final VC quality.

## Reproduction Conclusion

This reproduction successfully recovers the core behavior of the VoiceGrad DPM + BNF system.  
Under the correct dataset protocol, mel definition, BNF conditioning, diffusion setting, and vocoder configuration, the reproduced model reaches **the same performance level reported in the original paper** in closed-set evaluation.

The most important takeaway from this reproduction is that **paper-level performance depends not only on the diffusion model itself, but also on strict consistency across feature extraction, normalization, temporal alignment, inference setting, and vocoder design**.

## Core Files

- `dataset.py` — data loading, split logic, BNF alignment, cropping, normalization
- `model.py` — VoiceGrad network definition
- `diffusion.py` — cosine schedule, forward diffusion, reverse sampling
- `train.py` — training pipeline
- `config_16k.json` — HiFi-GAN configuration

## Notes

Some auxiliary files in the original project are retained for other upstream VC pipelines, but the final DPM reproduction mainly depends on:

- `dataset.py`
- `model.py`
- `diffusion.py`
- `train.py`
- `config_16k.json`

## Acknowledgement

This repository is a reproduction of the VoiceGrad paper for research and educational purposes.
