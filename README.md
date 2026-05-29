# Vehicle Detection IS — Detekcija vozila u saobraćaju

Projekat iz predmeta **Inteligentni sistemi** — detekcija objekata (automobili, pešaci, bicikli) iz vozila u pokretu pomoću bounding box-ova.

## 📁 Struktura projekta

```
├── data/
│   ├── raw/              # Originalni video snimci / slike
│   ├── annotations/      # Supervisely export (COCO JSON)
│   └── processed/        # Dataset podeljen na train/val/test
├── models/
│   ├── ssd_mobilenet.py  # Definicija SSD + MobileNetV3 modela
│   └── utils.py          # Pomoćne funkcije
├── notebooks/            # Jupyter notebook-ovi za analizu
├── docs/                 # Dokumentacija
│   ├── architecture_comparison.md
│   └── dataset_info.md
├── results/
│   ├── before_training/  # Predikcije pre treninga
│   ├── during_training/  # Predikcije tokom treninga
│   ├── after_training/   # Finalne predikcije
│   └── error_analysis/   # Analiza grešaka
├── videos/               # Video snimci za live inference
├── main.py               # Entry point — inference i demonstracija
├── train.py              # Trening modela
├── eval.py               # Evaluacija i metrike
├── visualize.py          # Vizuelizacija
├── pyproject.toml        # Konfiguracija i zavisnosti
└── README.md
```

## 🚀 Instalacija

### Preduslovi

- Python 3.12
- `uv` (menadžer paketa): `pip install uv`

### 1. Kloniranje repozitorijuma

```bash
git clone <repo-url>
cd inteligentni-sistemi
```

### 2. Instalacija zavisnosti

**🔴 AMD ROCm (Linux) — tvoja konfiguracija:**

```bash
uv sync --extra-index-url https://download.pytorch.org/whl/rocm6.4
```

**🔵 NVIDIA CUDA (Windows/Linux):**

```bash
uv sync
```

> PyTorch i TorchVision su definisani u `pyproject.toml`, ali se preuzimaju sa različitih
> indeksa u zavisnosti od hardvera. ROCm indeks sadrži `torch`, `torchvision` i
> `pytorch-triton-rocm` za AMD GPU. NVIDIA korisnici dobijaju CUDA verziju direktno sa PyPI.

### 3. Verifikacija

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'GPU dostupan: {torch.cuda.is_available()}')"
```

## 📊 Korišćene tehnologije

| Tehnologija | Svrha |
|---|---|
| **PyTorch 2.9 + ROCm 6.4** | Deep learning framework |
| **SSD + MobileNetV3** | Arhitektura modela za objektnu detekciju |
| **Supervisely** | Anotacija podataka (bounding box-ovi) |
| **pycocotools** | Evaluacija (mAP, IoU metrike) |
| **albumentations** | Augmentacija podataka |
| **TensorBoard** | Praćenje treninga |
| **OpenCV** | Obrada slika i video zapisa |

## 🎯 Klase

1. **Car** — automobili u saobraćaju
2. **Pedestrian** — pešaci
3. **Bicycle** — biciklisti (uključujući bicikl i osobu)

> ⚠️ Klasa Bicycle će biti zadržana samo ako ima dovoljno primera u datasetu.
