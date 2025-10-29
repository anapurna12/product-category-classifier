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
