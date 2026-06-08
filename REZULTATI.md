# 📊 Rezultati treninga - Detekcija objekata u saobraćaju

## 📁 Dataset

| Osobina | Vrednost |
|---------|----------|
| Slike | 401 (280 trening + 60 validacija + 61 test) |
| Klase | 3: Car, Pedestrian, Bicycle |
| Distribucija klasa | Car ~64%, Pedestrian ~31%, Bicycle ~5% |
| Format | YOLO (.txt label-e, 640×640 slike) |

---

## 📐 Objašnjenje metrika

| Metrika | Naziv | Šta znači |
|---------|-------|-----------|
| **Preciznost (P)** | Precision | **Kad model nešto detektuje - koliko je to tačno?** <br>80% = od 10 detekcija, 8 je tačno, 2 su lažne
| **Odziv (R)** | Recall | **Koliko stvarnih objekata je model pronašao?** <br>70% = od 10 stvarnih objekata, 7 je pronađeno, 3 propuštena - ovo bi trebalo da se poboljsa ako iskljucimo udaljene pesake iz dataseta
| **mAP50** | mean AP @ IoU=0.50 | **Glavna metrika!** Srednja preciznost za sve klase.<br>Računa se na pragu IoU=0.50 (dovoljno da kutija okvirno pogodi objekat)
| **mAP50-95** | mean AP @ IoU=0.50:0.95 | **Strožija metrika.** Prosek kroz više IoU pragova.<br>Kažnjava netačne bounding box-ove. Teže je postići visok broj.

> **IoU** (Intersection over Union) = koliko se predviđena kutija poklapa sa stvarnom. <br>IoU=0.50 = 50% poklapanja, IoU=0.95 = 95% poklapanja (skoro savršeno)
|---------|----------|
| Slike | 401 (280 trening + 60 validacija + 61 test) |
| Klase | 3: Car, Pedestrian, Bicycle |
| Distribucija klasa | Car ~64%, Pedestrian ~31%, Bicycle ~5% |
| Format | YOLO (.txt label-e, 640×640 slike) |

---

## 📈 Model 1: YOLOv8n (Nano) - osnovni

> **Fajl:** `modeli/yolov8/nano/...`
> **Parametri:** 3,006,233 (~3M)

### Metrike (na test skupu)

| Klasa | Instanci | Preciznost | Odziv | mAP50 | mAP50-95 |
|-------|:--------:|:----------:|:-----:|:-----:|:--------:|
| Car   | 126      | 87.4%      | 84.9% | 93.4% | 68.8%    |
| Ped   | 63       | 75.9%      | 49.2% | 63.0% | 30.0%    |
| Bicyc | 8        | 81.8%      | 75.0% | 83.6% | 53.7%    |
| **UKUPNO** | **197** | **81.7%** | **69.7%** | **80.0%** | **50.8%** |

### Tehnike balansiranja
- `cls=0.7` (povećan gubitak klasifikacije)
- `mixup=0.1` (mešanje slika)
- `copy_paste=0.1` (kopiranje objekata)

---

## 📈 Model 2: YOLOv8s (Small) - osnovni

> **Fajl:** `modeli/yolov8/small/osnovni_best.pt`
> **Parametri:** 11,126,253 (~11M)

### Metrike (na test skupu)

| Klasa      | Instanci | Preciznost | Odziv     | mAP50     | mAP50-95  |
|------------|:--------:|:----------:|:---------:|:---------:|:---------:|
| Car        | 126      | 88.1%      | 88.3%     | 94.2%     | 73.2%     |
| Pedestrian | 63       | 77.3%      | 54.0%     | 66.3%     | 35.0%     |
| Bicycle    | 8        | 85.6%      | 74.6%     | 84.9%     | 63.8%     |
| **UKUPNO** | **197**  | **83.7%**  | **72.3%** | **81.8%** | **57.4%** |

## Tumacenje grafika

- Ukupan mAP@0.5 (Srednja prosečna preciznost): 81.8%. Ovo ukazuje na to da model generalno ima vrlo dobre performanse.
- Optimalan prag sigurnosti (Confidence Threshold): Prema F1-Confidence krivoj, maksimalan balans između detektovanih   objekata i lažnih uzbuna postiže se pri pragu od 0.41 (41%). Ovo je preporučena vrednost parametra za implementaciju modela u praksi.
- Maksimalni Odziv (Recall): Čak i pri najnižem pragu sigurnosti, model uspeva da pronađe oko 90% svih objekata na slikama. Ostalih 10% model ne registruje uopšte

-Model je sklon laznim detekcijama na praznim delovima slike. Prijavio je 28 laznih automobila i 16 laznih pešaka tamo gde se zapravo nalazila samo pozadina (background). 

## Poredjenje klasa

- Klasa "Car" (Automobili) - mAP: 94.2%: Model radi izuzetno dobro na detekciji automobila. Visoko je pouzdan i promašuje veoma mali broj vozila (samo 13 u test setu).

- Klasa "Pedestrian" (Pešaci) - mAP: 66.3%: Ovo je najslabija tačka trenutnog modela. Imamo dva problema. Imamo 20 False Negative i 16 False Positive.

- Klasa "Bicycle" (Bicikli) - mAP: 84.9%: mAP je odlican, ali grafik je izuzetno stepenast, sto nas zajedno sa matricom konfuzija ukazuje na to da bicikala u test setu ima premalo. Medjutim, bicikala generalno ima u malom broju, tako da bi jedino poboljsanje moglo da se primeti prikupljanjem jos slika na kojima se pojavljuju.
  
## 📊 Poređenje modela

### Po mAP50

| Klasa                 | YOLOv8n Nano | YOLOv8s Small | Poboljšanje |
|-----------------------|:------------:|:-------------:|:-----------:|
| 🚗 Car mAP50          | 93.4%        | **94.2%**     | +0.8%       |
| 🚶 Pedestrian mAP50   | 63.0%        | **66.3%**     | **+3.3%**   |
| 🚲 Bicycle mAP50      | 83.6%        | **84.9%**     | +1.3%       |
| **UKUPNO mAP50**      | **80.0%**    | **81.8%**     | **+1.8%**   |
 
### Po Preciznosti i Odzivu

| Klasa                 | YOLOv8n P | YOLOv8n R | YOLOv8s P | YOLOv8s R |
|-----------------------|:---------:|:---------:|:---------:|:---------:|
| 🚗 Car                | 87.4%     | 84.9%     | **88.1%** | **88.3%** |
| 🚶 Pedestrian         | 75.9%     | 49.2%     | **77.3%** | **54.0%** |
| 🚲 Bicycle            | 81.8%     | 75.0%     | **85.6%** | 74.6%     |
| **UKUPNO**            | 81.7%     | 69.7%     | **83.7%** | **72.3%** |

> **P** = Preciznost, **R** = Odziv (Recall)

### Po mAP50-95 (strožija metrika)

| Klasa                   | YOLOv8n Nano | YOLOv8s Small | Poboljšanje |
|-------------------------|:------------:|:-------------:|:-----------:|
|-------------------------|:------------:|:-------------:|:-----------:|
| 🚗 Car mAP50-95         | 68.8%        | **73.2%**     | **+4.4%**   |
| 🚶 Pedestrian mAP50-95  | 30.0%        | **35.0%**     | **+5.0%**   |
| 🚲 Bicycle mAP50-95     | 53.7%        | **63.8%**     | **+10.1%**  |
| **UKUPNO mAP50-95**     | **50.8%**    | **57.4%**     | **+6.6%** 🏆|

### Zaključak
- YOLOv8s je **bolji u svim metrikama**
- Najveći skok je kod **preciznosti bounding box-ova** (mAP50-95 +6.6%)
- Pešaci su i dalje **najslabija karika** (66.3% mAP50) zbog nedovoljno primera, takodje model ne detektuje pesake koji su daleko. su pesaci koji se nalaze daleko sa strane puta i ne mozemo ih ugroziti tokom voznje, tako da cak i nije veliki problem, ali definitivno to utice na metrike za pesake
- Bicikli su **neočekivano dobri** (84.9%)- verovatno zato što su vizuelno izraziti

---

## 🤖 Poređenje sa pretreniranim modelom (yolo26n)

> **Pretrenirani yolo26n.pt** (5.3M parametara) - istreniran na COCO dataset-u (80 klasa)
> Za razliku od našeg modela koji je treniran **od nule**

> 📸 Vizuelna poređenja sačuvana u: `rezultati/poredjenje/`

### Zapažanja
- Pretrenirani model detektuje **više objekata**, ali to nisu objekti koji su nama od znacaja. Kola koja je detektovao su na velikoj razdaljini, to ni nije bio nas cilj. Pesaka kojeg je detektovao sa strane je pogresno oznacio.
- Naš model je **specjalizovan za naše 3 klase** i radi dobro uprkos malom broju slika
- Nas model radi sasvim dobro u poredjenju sa pretreniranim modelom.

---

## 🔮 Preporuke za poboljšanje

1. **Više podataka za pešake i bicikliste** - prikupiti još slika sa pešacima i biciklistima
2. **Druga arhitektura** (SSD, Faster R-CNN) - za poređenje
3. **Poboljsanje dataseta** - Mozda nije lose da pogledamo sta je model bio u stanju da detektuje, a da mi nismo oznacili, i onda da to takodje oznacimo na datasetu da ga ne bismo kaznjavali za neke dobre predikicije za koje nismo znali ni da bi bio sposoban. 
4. **Dodati mali broj slika na kojima nemamo nijedan objekat** - model mora da zna kako izgleda slika na kojoj nema objekta da ne bi oznacavao objekte tamo gde ih nema.