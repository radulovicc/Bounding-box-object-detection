"""
YOLOv8 trening - Detekcija objekata u saobracaju (Car, Pedestrian, Bicycle)

Pokretanje:
    uv run python src/treniraj.py
"""

import os
os.environ['ULTRALYTICS_LOGGING_LEVEL'] = 'ERROR'

import warnings
warnings.filterwarnings("ignore")

import sys
from io import StringIO
import logging
logging.getLogger("PIL").setLevel(logging.ERROR)
logging.getLogger("ultralytics").setLevel(logging.ERROR)


class PrigusivacStderr:
    def __enter__(self):
        self._stderr = sys.stderr
        sys.stderr = StringIO()
        return self
    def __exit__(self, *args):
        sys.stderr = self._stderr


from pathlib import Path
from ultralytics import YOLO


# === 1. LISTA SLIKA ZA PRACENJE NAPRETKA ===
SLIKE_ZA_PRACENJE = [
    "1141481_2. tura_20260525_162512",
    "1141481_2. tura_20260529_215342",
    "1141481_2. tura_20260525_163057",
    "1141481_2. tura_20260525_164748",
    "1141481_2. tura_20260525_173520",
    "1141481_2. tura_20260529_200007",
    "1141481_2. tura_20260529_202112",
    "1141481_2. tura_20260529_214638",
    "1141481_2. tura_20260529_220832",
    "1141481_2. tura_20260529_223002",
]


# === 2. CALLBACK ===
def sacuvaj_predikcije(trainer):
    """Svakih 10 epoha cuva predikcije za sve pracene slike."""
    trenutna_epoha = trainer.epoch + 1
    if trenutna_epoha % 10 != 0:
        return

    # Folder za slike napretka unutar ovog treninga
    progres_folder = Path(trainer.save_dir) / "progres_slike"
    progres_folder.mkdir(parents=True, exist_ok=True)

    putanja_modela = os.path.join(trainer.save_dir, 'weights', 'last.pt')
    if not os.path.exists(putanja_modela):
        return

    m = YOLO(putanja_modela)
    for naziv in SLIKE_ZA_PRACENJE:
        putanja_slike = f"dataset/images/testiranje/{naziv}.jpg"
        if not os.path.exists(putanja_slike):
            continue

        with PrigusivacStderr():
            r = m.predict(putanja_slike, device='cpu', verbose=False)

        folder_slike = progres_folder / naziv
        folder_slike.mkdir(parents=True, exist_ok=True)
        r[0].save(str(folder_slike / f"epoha_{trenutna_epoha}.jpg"))

    print(f"  [OK] Predikcije sacuvane za epohu {trenutna_epoha}")


# === 3. TRENING ===
print("=" * 52)
print("  TRENING MODELA - DETEKCIJA OBJEKATA")
print("=" * 52)
print("\n  Arhitektura: YOLOv8 Nano (3M parametara)")
print(f"  Pracenje napretka: {len(SLIKE_ZA_PRACENJE)} slika")
print("  Snima se svakih 10 epoha unutar foldera treninga")
print("\n  Pokrecem trening od 200 epoha...")

IME_TRENINGA = 'yolov8n_osnovni'

model = YOLO('yolov8n.yaml')
model.add_callback("on_fit_epoch_end", sacuvaj_predikcije)

with PrigusivacStderr(): #Sluzi da malo rastereti terminal od prevelike kolicine informacija tokom treninga
    model.train(
        data='dataset/podesavanja.yaml',
        epochs=200,
        imgsz=640,
        batch=16,
        device=0,
        degrees=0.0, fliplr=0.5, scale=0.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, #Hue, Saturation, Value (Nijansa, Zasićenost, Osvetljenost)
        #Hue nam sluzi da slike dobiju razlicite nijanse boje, kako model ne bi naucio sablon po boji ako imamo previse kola iste boje
        #Saturation sluzi da menja sliku iz crno-bele u jarke boje posto kamere cesto mogu da budu loseg kvaliteta
        #Value sluzi da menja osvetljenost slike, znaci neke slike koje su slikane danju mogu da deluju kao da su slikane nocu
        translate=0.1, #ovo dozvoljava da se slike transliraju u bilo kom pravcu do 10%, dobra je stvar jer nece nas auto uvek biti na toj pozociji iz koje smo slikali pa je dobro radi raznovrsnosti
        project='napredak',
        name=IME_TRENINGA,
        verbose=True, #Da bi tokom treninga nesto i bilo ispisano na konzoli od informacija za model
    )

print()
print("  TRENING ZAVRSEN!")

# === 4. KORISTIMO trainer.save_dir (YOLO cuva u runs/detect/napredak/...) ===
poslednji = str(model.trainer.save_dir)
print(f"\n  {'=' * 52}")
print(f"  KONACNE METRIKE")
print(f"  {'=' * 52}")

putanja_best = os.path.join(poslednji, 'weights', 'best.pt')
if os.path.exists(putanja_best):
    with PrigusivacStderr():
        ev = YOLO(putanja_best)
        rez = ev.val(data='dataset/podesavanja.yaml', verbose=False)

print(f"  {'─' * 48}")
print(f"  {'Klasa':<14}{'Preciznost':<14}{'Odziv':<12}{'mAP50':<10}")
print(f"  {'─' * 48}")

imena = ['Car', 'Pedestrian', 'Bicycle']
for i, ime in enumerate(imena):
    print(f"  {ime:<14}{rez.box.p[i]:<13.1%} {rez.box.r[i]:<11.1%} {rez.box.ap50[i]:<8.1%}")

print(f"  {'─' * 48}")
print(f"  {'UKUPNO':<14}{rez.box.p.mean():<13.1%} {rez.box.r.mean():<11.1%} {rez.box.map50:<8.1%}")
print(f"  {'=' * 52}")

# === 5. KOPIRAJ MODEL U modeli/ ===
import shutil
model_odrediste = Path("modeli/yolov8/nano/osnovni_best.pt")
model_odrediste.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(putanja_best, str(model_odrediste))
print(f"  Model kopiran u: {model_odrediste}")

print(f"\n  Rezultati: {poslednji}/")
print(f"  Slike napretka: {poslednji}/progres_slike/")
print(f"  Za detaljniju evaluaciju: uv run python src/evaluiraj.py")
