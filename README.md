# Vehicle Detection IS — Detekcija vozila u saobraćaju

Projekat iz predmeta **Inteligentni sistemi** — detekcija objekata (automobili, pešaci, bicikli) iz vozila u pokretu pomoću bounding box-ova.

## 📁 Struktura projekta

```
inteligentni-sistemi/
├── src/                          # Sav izvorni kod
│   ├── treniraj_yolov8n.py       # YOLOv8 Nano (3M param.)
│   ├── treniraj_yolov8n_balansirano.py  # YOLOv8n + Focal Loss za balans klasa
│   ├── treniraj_yolov8s.py       # YOLOv8 Small (11M param.)
│   ├── evaluiraj.py              # Evaluacija modela
│   └── podeli_dataset.py         # Podela slika na trening/validaciju/test
├── dataset/
│   ├── slike/
│   │   ├── treniranje/    # 280 slika
│   │   ├── validacija/    # 60 slika
│   │   └── testiranje/    # 61 slika
│   ├── labels/            # YOLO format labela (.txt)
│   │   ├── treniranje/
│   │   ├── validacija/
│   │   └── testiranje/
│   └── podesavanja.yaml   # YOLO data config
├── modeli/                # Istrenirani modeli
│   ├── yolov8/
│   │   ├── nano/          # YOLOv8 Nano modeli
│   │   └── small/         # YOLOv8 Small modeli
│   ├── pretrenirani/      # Pretrenirani .pt fajlovi
│   └── drugi_modeli/      # Za buduće arhitekture (SSD, ...)
├── napredak/              # Slike napretka tokom treninga
│   ├── trening_1/
│   └── trening_2/
├── .venv/                 # Virtuelno okruzenje
├── pyproject.toml
├── README.md
└── .gitignore
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
| **YOLOv8 (Nano/Small)** | Arhitektura modela za objektnu detekciju |
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
