# Product Category Classifier

## Opis
Cilj projekta je da automatski predvidi kategoriju proizvoda (npr. "mobile phones", "washing machines", "digital cameras") na osnovu naslova proizvoda.

## Struktura projekta
- data/IMLP4_TASK_03-products.csv  — skup podataka
- src/train_model.py               — trenira model i snima ga u models/product_category_model.pkl
- src/predict_category.py          — koristi sačuvani model da predvidi kategoriju za novi naslov
- models/product_category_model.pkl — fajl sa modelom (posle treniranja)
- requirements.txt                 — zavisnosti

## Kako se koristi (ukratko)
1. Instaliraju se zavisnosti iz requirements.txt.
2. Pokrene se train_model.py da istrenira model.
3. Onda se pokrene predict_category.py da se dobije kategorija za novi naslov proizvoda.

## Zaključak

Ovaj projekat rešava zadatak automatske kategorizacije proizvoda na osnovu naslova proizvoda.

Glavne faze:
1. Učitavanje podataka iz `IMLP4_TASK_03-products.csv`.
2. Čišćenje kategorija (spajanje sličnih labela, npr. "mobile phone" -> "mobile phones").
3. Pretvaranje teksta (`Product Title`) u numeričke osobine pomoću TF-IDF n-gramova.
4. Dodavanje dodatnih ručnih osobina iz naslova (broj reči, dužina naslova, prisustvo cifara itd.).
5. Treniranje LinearSVC klasifikatora.
6. Evaluacija modela (accuracy i classification report na test skupu).
7. Snimanje istreniranog pipeline-a u `models/product_category_model.pkl`.
8. Predikcija kategorije za nove proizvode korišćenjem `src/predict_category.py`.

Ovo znači da onboarding novog proizvoda u webshop može da predloži kategoriju automatski, umesto da neko ručno bira kategoriju.

