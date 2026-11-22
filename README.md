# Empirical Evaluation of Knowledge Distillation Paradigms Across Lightweight Architectures for Crop–Weed Semantic Segmentation

Official reconstruction code, stratified splits (`SPLIT_OPT`), and **12 pre-trained student checkpoints** accompanying the manuscript *(under review)*:

> **Indra Agustian, Igi Ardiyanto, Sunu Wibirama, and Ruvita Faurina**, *"Empirical Evaluation of Knowledge Distillation Paradigms Across Lightweight Architectures for Crop–Weed Semantic Segmentation"*.

---

## Repository Structure

```text
├── checkpoints/                 # 12 pre-trained last-EMA checkpoints (EfficientViT-B0 & Fast-SCNN)
├── losses/
│   └── sota_kd_losses.py        # Implementations of Vanilla KL, AT, CWD, MGD, DIST, IFVD
├── models/
│   ├── efficientvit_b0.py       # EfficientViT-B0 (0.405M params)
│   ├── modelfast_scnn.py        # Fast-SCNN (1.130M params)
│   └── modelteacher_DMiTB.py    # DeepLabV3+ (MiT-B3 + ExG) Teacher (51.83M params)
├── results/
│   └── cwfid_6sota_kd_results.csv
├── splits/
│   └── split_stratified.json    # Quantile-L1 stratified CWFID split (45 train / 6 val / 9 test)
├── eval_checkpoints.py          # Evaluate all 12 pre-trained checkpoints on CWFID test set
├── train_kd.py                  # Train any (student, kd_method) pair from scratch
└── requirements.txt
```

---

## Benchmark Summary (CWFID Held-Out Test Set, $N=9$, $512 \times 512$)

| Student (`Jetson 5W`) | Method | mIoU (%) | Crop IoU (%) | Weed IoU (%) | Weed Prec (%) | Weed Rec (%) | $\Delta$mIoU (pp) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EfficientViT-B0** (`22.89 FPS`, `184.0 mJ`) | No-KD | 77.56 | 61.60 | 73.29 | — | — | 0.00 |
| | **Vanilla KL** | **80.09** | 66.71 | **75.65** | **84.46** | 87.88 | **+2.53** |
| | IFVD | 79.97 | **68.18** | 73.97 | 79.29 | **91.68** | +2.41 |
| | DIST | 79.59 | 66.18 | 74.68 | 82.44 | 88.80 | +2.03 |
| | MGD | 78.85 | 64.70 | 74.00 | 79.48 | 91.48 | +1.28 |
| | AT | 76.62 | 61.63 | 70.79 | 78.46 | 87.87 | -0.94 |
| | CWD | 75.81 | 55.52 | 74.33 | 80.39 | 90.79 | -1.75 |
| **Fast-SCNN** (`30.92 FPS`, `126.0 mJ`) | No-KD | 75.41 | 57.91 | 70.65 | — | — | 0.00 |
| | **DIST** | **79.22** | **65.18** | **74.55** | 82.85 | 88.15 | **+3.81** |
| | MGD | 78.39 | 63.89 | 73.50 | 79.08 | **91.23** | +2.98 |
| | CWD | 76.95 | 61.21 | 72.00 | 78.11 | 90.20 | +1.54 |
| | IFVD | 76.51 | 61.55 | 70.63 | 76.62 | 90.03 | +1.10 |
| | Vanilla KL | 74.88 | 56.12 | 70.68 | **83.00** | 82.64 | -0.54 |
| | AT | 71.51 | 53.04 | 64.52 | 73.80 | 83.69 | -3.90 |

---

## Quick Start

### 1. Install Dependencies & Download CWFID
```bash
pip install -r requirements.txt
wget https://github.com/cwfid/dataset/archive/v1.0.tar.gz -O cwfid_v1.0.tar.gz
tar -zxf cwfid_v1.0.tar.gz
```

### 2. Evaluate All 12 Pre-Trained Checkpoints
```bash
python eval_checkpoints.py --data_root ./dataset-1.0
```

### 3. Reconstruct Training from Scratch
```bash
python train_kd.py \
  --data_root ./dataset-1.0 \
  --teacher_ckpt /path/to/teacher_scratch_lossc_s42_last_ema.pth \
  --student EfficientViT_B0 \
  --kd_method Vanilla_KL \
  --steps 5000 \
  --seed 42
```

---

## Citation
The accompanying manuscript is currently **under review**. If you use this benchmark, split, or pre-trained checkpoints in your research, please cite this GitHub repository:

```bibtex
@misc{agustian2026empiricalkd,
  author       = {Agustian, Indra and Ardiyanto, Igi and Wibirama, Sunu and Faurina, Ruvita},
  title        = {Empirical Evaluation of Knowledge Distillation Paradigms Across Lightweight Architectures for Crop--Weed Semantic Segmentation},
  year         = {2026},
  howpublished = {\url{https://github.com/indraagustian/cwfid-lightweight-kd}},
  note         = {GitHub repository; manuscript under review}
}
```
