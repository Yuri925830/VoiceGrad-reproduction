# VoiceGrad (DPM + BNF)
### Research-Grade Reproduction of Diffusion Probabilistic Voice Conversion with Bottleneck Feature Conditioning

<p align="center">

![Task](https://img.shields.io/badge/Task-Non--Parallel%20Any--to--Many%20Voice%20Conversion-blue)
![Paradigm](https://img.shields.io/badge/Paradigm-Score--Based%20Generative%20Modeling-black)
![Backbone](https://img.shields.io/badge/Framework-Diffusion%20Probabilistic%20Modeling-purple)
![Conditioning](https://img.shields.io/badge/Conditioning-144D%20BNF-darkgreen)
![Acoustic](https://img.shields.io/badge/Acoustic%20Space-80D%20Log--Mel-orange)
![Status](https://img.shields.io/badge/Status-Paper--Level%20Reproduced-success)

</p>

> This repository presents a high-fidelity reproduction of the **DPM + BNF** variant of **VoiceGrad**, a **non-parallel any-to-many voice conversion** framework built upon **Diffusion Probabilistic Models**, **score matching**, and **annealed Langevin dynamics**.  
> The reproduced system reconstructs the complete research pipeline spanning **acoustic representation design**, **BNF-conditioned diffusion training**, **speaker-conditional reverse denoising**, **mel-vocoder consistency**, **closed-set protocol restoration**, and **objective evaluation at paper level**.

---

## Abstract

VoiceGrad formulates voice conversion as a **score-based iterative denoising process in acoustic feature space**.  
Instead of relying on parallel supervision, the model learns a **speaker-conditional score estimator** over noisy speech representations and progressively transports source speech features toward the target-speaker manifold under a **reverse diffusion trajectory**.

This reproduction targets the **DPM + BNF** setting described in the original paper and reconstructs the full operational stack:

- **Diffusion Probabilistic Modeling (DPM)**
- **score-function estimation**
- **annealed Langevin dynamics**
- **144-dimensional Bottleneck Feature (BNF) conditioning**
- **80-dimensional log-mel acoustic modeling**
- **U-Net-style fully convolutional denoising backbone**
- **speaker embedding control**
- **cosine-scheduled diffusion process**
- **intermediate-level reverse-process initialization**
- **self-trained 16 kHz HiFi-GAN waveform synthesis**
- **closed-set CMU ARCTIC evaluation**

The final reproduced model reaches the **performance level reported in the VoiceGrad paper** under the closed-set protocol.

---

## Highlights

- **Faithful reproduction of VoiceGrad DPM + BNF**
- **Paper-level closed-set evaluation performance**
- **End-to-end reconstruction from preprocessing to waveform synthesis**
- **BNF-conditioned score-based acoustic generation**
- **Cosine-noise diffusion training with iterative reverse denoising**
- **HiFi-GAN-coupled mel feature pipeline**
- **Metric-driven checkpoint analysis across CER / MCD / LFC / pMOS**
- **Research-oriented codebase for diffusion-based VC study, analysis, and extension**

---

## Method Overview

The reproduced system performs voice conversion through a **multi-stage stochastic refinement process** rather than direct frame-to-frame regression.

### Core formulation
- **Input condition:** source linguistic representation in **144D BNF space**
- **Acoustic target space:** **80D log-mel spectrogram**
- **Generative paradigm:** **score-based diffusion modeling**
- **Reverse process:** iterative denoising toward the **target-speaker acoustic distribution**
- **Control signal:** explicit **speaker embedding conditioning**
- **Waveform decoder:** **HiFi-GAN**

### Conversion pipeline

```text
raw waveform
  -> 80D log-mel extraction
  -> global mel statistics estimation
  -> 144D BNF extraction
  -> BNF / mel temporal synchronization
  -> speaker-conditional diffusion training
  -> reverse denoising conversion
  -> mel restoration
  -> neural vocoder synthesis
  -> objective evaluation
