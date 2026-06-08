"""
Cista evaluacija modela - samo bitne metrike, bez tehnickog suma.

Pokretanje:
    uv run python src/evaluiraj.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
import logging
from pathlib import Path
from io import StringIO

logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)


class PrigusivacStderr:
    def __enter__(self):
        self._stderr = sys.stderr
        sys.stderr = StringIO()
        return self
    def __exit__(self, *args):
        sys.stderr = self._stderr


from ultralytics import YOLO


def proceni(vrednost, prag_dobar=0.70, prag_odlican=0.85):
    if vrednost >= prag_odlican:
        return "Odlican"
    elif vrednost >= prag_dobar:
        return "Dobar"
    elif vrednost >= 0.50:
        return "Osrednji"
    else:
        return "Slab"


def main():
    # Trazi modele u modeli/ folderu
    modeli = sorted(Path("modeli").rglob("*_best.pt"))
    if not modeli:
        print("Nema treniranog modela. Pokreni: uv run python src/treniraj_yolov8n.py")
        return

    # Izlistaj sve modele da korisnik vidi sta je dostupno
    print("Dostupni modeli:")
    for i, m in enumerate(modeli):
        print(f"  [{i}] {m.relative_to('modeli')}")
    print()

    # Uzmi poslednji (najnoviji) ili pusti korisnika da izabere
    izbor = input("Izaberi broj modela (Enter za poslednji): ").strip()
    if izbor == "":
        izbor_idx = -1
    else:
        izbor_idx = int(izbor)

    putanja_modela = str(modeli[izbor_idx])
    naziv_modela = str(modeli[izbor_idx].relative_to('modeli'))

    print("=" * 55)
    print("  EVALUACIJA MODELA - DETEKCIJA OBJEKATA")
    print("=" * 55)
    print(f"  Model: {naziv_modela}")
    print()

    with PrigusivacStderr():
        model = YOLO(putanja_modela)
        rez = model.val(data='dataset/podesavanja.yaml', verbose=False)

    # UKUPNE METRIKE
    p_uk = rez.box.p.mean()
    r_uk = rez.box.r.mean()
    m50 = rez.box.map50
    m95 = rez.box.map

    print("  UKUPNE METRIKE:")
    print(f"  {'─' * 42}")
    print(f"  {'Metrika':<20} {'Vrednost':<10} {'Ocena'}")
    print(f"  {'─' * 42}")
    print(f"  {'Preciznost':<20} {p_uk:<8.1%}  {proceni(p_uk, 0.65, 0.80)}")
    print(f"  {'Odziv (Recall)':<20} {r_uk:<8.1%}  {proceni(r_uk, 0.65, 0.80)}")
    print(f"  {'mAP@50':<20} {m50:<8.1%}  {proceni(m50)}")
    print(f"  {'mAP@50:95':<20} {m95:<8.1%}  {proceni(m95, 0.40, 0.55)}")
    print()

    # PO KLASAMA
    print("  PO KLASAMA:")
    print(f"  {'─' * 56}")
    print(f"  {'Klasa':<14} {'Slike':<7} {'Objekti':<9} {'Preciznost':<12} {'Odziv':<10} {'mAP50':<10}")
    print(f"  {'─' * 56}")

    imena = ['Car', 'Pedestrian', 'Bicycle']
    for i, ime in enumerate(imena):
        p = rez.box.p[i]
        r_val = rez.box.r[i]
        ap50 = rez.box.ap50[i]
        br_s = rez.nt_per_image[i]
        br_o = rez.nt_per_class[i]
        print(f"  {ime:<14} {br_s:<7} {br_o:<9} {p:<10.1%}  {r_val:<8.1%}  {ap50:<8.1%}")

    print(f"  {'─' * 56}")
    print(f"  {'UKUPNO':<14} {sum(rez.nt_per_image):<7} {sum(rez.nt_per_class):<9} {p_uk:<10.1%}  {r_uk:<8.1%}  {m50:<8.1%}")
    print()

    # BRZINA
    print(f"  BRZINA:")
    print(f"     Vreme inferencije: {rez.speed['inference']:.1f} ms po slici")
    print(f"     (~ {1000/rez.speed['inference']:.0f} slika/sekundi)")
    print()

    # ZAKLJUCAK
    print("  ZAKLJUCAK:")
    if m50 >= 0.70:
        print(f"     Model je solidan (mAP50 = {m50:.1%}).")
    else:
        print(f"     Model ima prostora za poboljsanje (mAP50 = {m50:.1%}).")
    najbolja = imena[rez.box.ap50.argmax()]
    najslabija = imena[rez.box.ap50.argmin()]
    print(f"     Najbolje detektuje: {najbolja}")
    print(f"     Najslabije detektuje: {najslabija}")
    print(f"     Uporedi sa: uv run python src/evaluiraj.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
