"""
Skripta za podelu dataset-a na train / val / test.

Supervisely je eksportovao sve podatke u train folder (slike i labele).
Ova skripta ih deli u YOLO strukturu:

data/
├── images/
│   ├── train/  (70%)
│   ├── val/    (15%)
│   └── test/   (15%)
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data_config.yaml

"""

import os
import shutil
import random
from pathlib import Path

# ─── Konfiguracija ───────────────────────────────────────────────
SOURCE_IMAGES = "data/processed/images/train"
SOURCE_LABELS = "data/processed/labels/train"
DEST = "data"

SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42

# YOLO klase (iz data_config.yaml)
CLASS_NAMES = ["Car", "Pedestrian", "Bicycle"]
CLASS_COLORS = [[126, 211, 33], [208, 2, 27], [98, 0, 255]]
NC = len(CLASS_NAMES)
# ─────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("Podela dataset-a na train / val / test")
    print("=" * 60)

    # 1. Proveri da li izvorni folderi postoje
    if not os.path.isdir(SOURCE_IMAGES):
        print(f"[GREŠKA] Izvorni folder sa slikama ne postoji: {SOURCE_IMAGES}")
        return

    if not os.path.isdir(SOURCE_LABELS):
        print(f"[GREŠKA] Izvorni folder sa labelama ne postoji: {SOURCE_LABELS}")
        return

    # 2. Uzmi sva imena fajlova (bez ekstenzije)
    image_files = sorted([
        f for f in os.listdir(SOURCE_IMAGES)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if not image_files:
        print("[GREŠKA] Nema slika u izvornom folderu!")
        return

    # Proveri da li za svaku sliku postoji labela
    names = []
    missing_labels = []
    for img_file in image_files:
        stem = Path(img_file).stem
        label_file = stem + ".txt"
        if os.path.isfile(os.path.join(SOURCE_LABELS, label_file)):
            names.append(stem)
        else:
            missing_labels.append(img_file)

    original_count = len(image_files)
    valid_count = len(names)

    print(f"\nPronađeno slika:          {original_count}")
    print(f"Sa odgovarajućom labelom:  {valid_count}")

    if missing_labels:
        print(f"[UPOZORENJE] {len(missing_labels)} slika nema labele:")
        for m in missing_labels[:5]:
            print(f"  - {m}")
        if len(missing_labels) > 5:
            print(f"  ... i još {len(missing_labels) - 5}")

    if valid_count == 0:
        print("[GREŠKA] Nema validnih parova slika-labela!")
        return

    # 3. Promešaj i podeli
    random.seed(RANDOM_SEED)
    random.shuffle(names)

    total = len(names)
    train_end = int(total * SPLIT_RATIOS["train"])
    val_end = train_end + int(total * SPLIT_RATIOS["val"])

    splits = {
        "train": names[:train_end],
        "val": names[train_end:val_end],
        "test": names[val_end:],
    }

    print(f"\nPodela ({SPLIT_RATIOS['train']*100:.0f}/{SPLIT_RATIOS['val']*100:.0f}/{SPLIT_RATIOS['test']*100:.0f}):")
    for split_name, split_names in splits.items():
        print(f"  {split_name}: {len(split_names)} primera")

    # 4. Kreiraj foldere i prebaci fajlove
    print("\nPrebacivanje fajlova...")
    for split_name, file_names in splits.items():
        img_dir = Path(DEST) / "images" / split_name
        lbl_dir = Path(DEST) / "labels" / split_name

        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        moved = 0
        for name in file_names:
            # Odredi ekstenziju slike
            for ext in ['.jpg', '.jpeg', '.png']:
                src_img = os.path.join(SOURCE_IMAGES, name + ext)
                if os.path.isfile(src_img):
                    break
            else:
                print(f"[UPOZORENJE] Slika za '{name}' nije pronađena!")
                continue

            src_lbl = os.path.join(SOURCE_LABELS, name + ".txt")

            if not os.path.isfile(src_lbl):
                print(f"[UPOZORENJE] Labela za '{name}' nije pronađena!")
                continue

            # Premeštaj
            shutil.move(src_img, str(img_dir / (name + ext)))
            shutil.move(src_lbl, str(lbl_dir / (name + ".txt")))
            moved += 1

        print(f"  ✓ {split_name}: prebačeno {moved} primera")

    # 5. Proveri da li je source folder prazan
    remaining = [f for f in os.listdir(SOURCE_IMAGES)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if remaining:
        print(f"\n[INFO] U source folderu je ostalo {len(remaining)} slika (verovatno bez labela)")

    # 6. Napravi data_config.yaml
    config_content = f"""# YOLO Data Config
# Automatski generisano - podela dataset-a
# Train: {len(splits['train'])} primera
# Val:   {len(splits['val'])} primera
# Test:  {len(splits['test'])} primera

train: ../data/images/train
val: ../data/images/val
test: ../data/images/test

nc: {NC}
names: {CLASS_NAMES}
"""
    config_path = Path(DEST) / "data_config.yaml"
    with open(config_path, "w") as f:
        f.write(config_content)
    print(f"\n✅ Kreiran config: {config_path}")

    # 7. Sažetak
    print("\n" + "=" * 60)
    print("✅ Podela završena!")
    print("=" * 60)
    print(f"\nKonačna struktura:")
    for split_name in ["train", "val", "test"]:
        img_count = len(os.listdir(Path(DEST) / "images" / split_name))
        lbl_count = len(os.listdir(Path(DEST) / "labels" / split_name))
        print(f"  {split_name}: {img_count} slika, {lbl_count} labela")


if __name__ == "__main__":
    main()
