# UltraLBM-UNet

This repository provides the official PyTorch implementation of **UltraLBM-UNet**, an ultra-lightweight bidirectional Mamba-based U-Net for skin lesion segmentation.

arXiv: https://www.arxiv.org/abs/2512.21584

---

## Environments

The code is tested under the following environment:

```
Python        : 3.12.12
PyTorch       : 2.8.0+cu126
Torchvision   : 0.23.0+cu126
CUDA          : 12.6
```

### Dependencies

```bash
pip install git+https://github.com/state-spaces/mamba.git@v2.2.2
```

---

## Dataset

All datasets should be placed under the `data/` directory.

```
data/
├── ISIC/
│   ├── isic17/
│   └── isic18/
└── PH2/
```

### ISIC 2017 and ISIC 2018

ISIC 2017 and ISIC 2018 are used as training datasets.  
The dataset organization follows the preprocessing protocol of [MALUNet](https://github.com/JCruan519/MALUNet).

```
isic17/ (or isic18/)
├── train/
│   ├── images/
│   │   └── *.png
│   └── masks/
│       └── *.png
└── val/
    ├── images/
    │   └── *.png
    └── masks/
        └── *.png
```

---

### PH<sup>2</sup> (External Validation Dataset)

PH<sup>2</sup>
 is used only as an external validation dataset to evaluate generalization performance.  
No PH<sup>2</sup> images are used during training.

```
PH2/
└── val/
    ├── images/
    │   └── *.png
    └── masks/
        └── *.png
```

---

### Dataset Download

All datasets have been packaged together and can be downloaded from [data](https://drive.google.com/file/d/1xicSLTkxghgdbHE12znwajXHfAS5PZCm/view?usp=drive_link)

After downloading, extract the archive and place the contents under the `data/` directory so that the structure matches the examples above.

---

## Training

To train UltraLBM-UNet from scratch, run:

```bash
python train.py
```
---

## Distillation Training

To train the ultra-compact student model using the proposed boundary-focused hybrid distillation strategy, run:

```bash
python distillation_train_boundary_focused_fixed.py \
    --teacher_path <path_to_teacher_model.pth>
```

Replace `<path_to_teacher_model.pth>` with the path to the pretrained UltraLBM-UNet checkpoint used as the teacher model.

Distillation results are saved in the `results/` directory.

---

## Testing

To evaluate a trained model, run:

```bash
python test.py --work_dir <path_to_results_folder>
```

Replace `<path_to_results_folder>` with the directory containing the trained model checkpoints.

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{fan2025ultralbmunetultralightbidirectionalmambabased,
      title={UltraLBM-UNet: Ultralight Bidirectional Mamba-based Model for Skin Lesion Segmentation},
      author={Linxuan Fan and Juntao Jiang and Weixuan Liu and Zhucun Xue and Jiajun Lv and Jiangning Zhang and Yong Liu},
      year={2025},
      eprint={2512.21584},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.21584}
}
```
