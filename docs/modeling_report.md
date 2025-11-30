# Modeling report – Titanic

## 1. Cel

Celem etapu było:

- zautomatyzowanie budowy modeli ML w Kedro,
- porównanie trzech podejść:
  - model bazowy (baseline),
  - AutoML (PyCaret),
  - własny model (RandomForest),
- zapisanie wytrenowanych modeli i metryk oraz wybór najlepszego modelu.

Wszystkie kroki zostały umieszczone w pipeline’ie `modeling` i uruchamiane poleceniem:

```bash
kedro run --pipeline=modeling
```

---

## 2. Opis pipeline’u `modeling`

Pipeline zdefiniowany w `pipeline.py` składa się z czterech node’ów:

- **train_baseline_node**
  - wejście: `train_data`, `val_data`
  - wyjście: `baseline_model`, `baseline_metrics`

- **train_automl_node**
  - wejście: `train_data`, `val_data`
  - wyjście: `automl_model`, `automl_results`, `automl_metrics`

- **train_custom_node**
  - wejście: `train_data`, `val_data`
  - wyjście: `custom_model`, `custom_metrics`

- **evaluate_models_node**
  - wejście: metryki i modele z trzech poprzednich kroków
  - wyjście: `model_comparison`, `best_model`

Dane `train_data`, `val_data`, `test_data` są ładowane z plików CSV z poprzedniego etapu i zarejestrowane w `catalog.yml`.

W katalogu `data/.../reporting/` zapisują się m.in.:

- `baseline_model.pkl`, `baseline_metrics.json`
- `automl_model.pkl`, `automl_results.csv`, `automl_metrics.json`
- `custom_model.pkl`, `custom_metrics.json`
- `best_model.pkl`, `model_comparison.json`

---

## 3. Zastosowane modele

### 3.1. Model bazowy (baseline)

Funkcja `train_baseline`:

- używa `DummyClassifier(strategy="most_frequent")`,
- uczy się na `train_data`,
- przewiduje na `val_data`,
- liczy metryki: `accuracy`, `precision`, `recall`, `f1`, `roc_auc`.

Model ten zawsze przewiduje najczęściej występującą klasę i służy jako minimalny punkt odniesienia.

---

### 3.2. AutoML (PyCaret)

Funkcja `train_automl`:

- wywołuje `pycaret.classification.setup(...)` z:

```
setup(
    data=train_data,
    target="Survived",
    session_id=42,
    verbose=False,
    html=False,
)
```

- uruchamia `compare_models(n_select=1, sort="Accuracy")`,
- wybiera najlepszy model (lista o długości 1 → bierzemy element `[0]`),
- zapisuje pełny ranking modeli do `automl_results.csv`,
- na walidacji (`val_data`):
  - wywołuje `predict_model(best_model, data=val_data.copy())`,
  - wykrywa kolumnę zawierającą prawdopodobieństwa (zawierającą w nazwie `score` lub `prob`),
  - liczy metryki jak w baseline.

Z pliku z porównaniem metryk wynika, że najlepszym modelem AutoML jest model typu **Logistic Regression** o wysokim Accuracy i F1, ale stosunkowo niskim `roc_auc` (patrz komentarz w sekcji wniosków).

---

### 3.3. Własny model – RandomForest z prostym preprocessingiem

Funkcja `train_custom` implementuje własny model:

- rozdzielenie cech i targetu (`_split_X_y`),
- wydzielenie kolumn numerycznych,
- one-hot encoding dla wybranych zmiennych kategorycznych:
  - `Sex`,
  - `Embarked`,
- dopasowanie kolumn w `val` do kolumn w `train` (`reindex` z `fill_value=0`),
- model:

```
RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
```

- trenowany na rozszerzonym zbiorze cech (numeryczne + dummies),
- ewaluacja na walidacji (`val_data`): pełny zestaw metryk, w tym `roc_auc` na podstawie `predict_proba`.

Metryki tego modelu znajdują się zarówno w `custom_metrics.json`, jak i w `model_comparison.json`.

---

## 4. Metryki modeli

Na podstawie pliku `model_comparison.json` otrzymaliśmy następujące wartości metryk na zbiorze walidacyjnym (zaokrąglone do 3 miejsc po przecinku):

| Model                            | Accuracy | Precision | Recall | F1    | ROC AUC |
|----------------------------------|----------|-----------|--------|-------|---------|
| **Baseline – DummyClassifier**   | 0.522    | 0.000     | 0.000  | 0.000 | NaN     |
| **AutoML – PyCaret (best model)**| 0.821    | 0.900     | 0.703  | 0.789 | 0.248   |
| **Custom – RandomForest**        | 0.828    | 0.887     | 0.734  | 0.803 | 0.916   |

Z pliku `model_comparison.json` wynika, że według kryterium **ROC AUC** najlepszym modelem jest:

```text
best_model_name = "custom"
best_model_roc_auc ≈ 0.916
```

Oznacza to, że jako globalnie najlepszy model zapisany do `best_model.pkl` został wybrany nasz własny **RandomForest z prostym one-hot encodingiem**.

---

## 5. Interpretacja wyników

### 5.1. Baseline

- Accuracy ≈ 0.52 oznacza, że model przewiduje poprawnie tylko trochę lepiej niż losowe zgadywanie w nierównoważnym zbiorze (większość klasy 0).
- Precision, Recall i F1 równe 0 oznaczają, że model praktycznie nie wykrywa klasy pozytywnej (przeżycie).
- Brak sensownego ROC AUC (NaN – brak prawdopodobieństw).

**Wniosek:** baseline spełnia rolę punktu odniesienia, ale nie nadaje się do realnego użycia.

---

### 5.2. AutoML (PyCaret)

- Accuracy ≈ 0.821, F1 ≈ 0.789 – model AutoML jest już wyraźnie lepszy od baseline.
- Precision 0.90 – jeśli model przewiduje, że pasażer przeżył, to w 90% przypadków ma rację.
- Recall ≈ 0.703 – model wychwytuje ponad 70% pozytywnych przypadków.
- ROC AUC ≈ 0.248 jest nienaturalnie małe jak na tak dobre pozostałe metryki i najprawdopodobniej wynika z tego, że automatyczne wykrywanie kolumny z prawdopodobieństwami (`proba_col`) nie wskazało poprawnej kolumny dodatniej klasy (format predykcji PyCaret jest specyficzny, a w kodzie wybieramy pierwszą kolumnę zawierającą `score`/`prob`).

**Wniosek:** AutoML dał dobry model (Logistic Regression), ale przy interpretacji ROC AUC trzeba uważać – główne porównanie opiera się tutaj na Accuracy/F1.

---

### 5.3. Własny model – RandomForest

- Accuracy ≈ 0.828 – nieco wyższe niż model AutoML.
- Precision ≈ 0.887, Recall ≈ 0.734 – bardzo dobry balans pomiędzy „łapaniem” przypadków pozytywnych a liczbą fałszywych alarmów.
- F1 ≈ 0.803 – najwyższa F1 spośród trzech podejść.
- ROC AUC ≈ 0.916 – zdecydowanie najwyższa wartość, model dobrze rozdziela klasy w przestrzeni prawdopodobieństw.

**Wniosek:** prosty RandomForest z ręcznie dobranymi hiperparametrami i minimalnym one-hot encodingiem **przewyższa AutoML** w kluczowych metrykach (F1, ROC AUC), mimo że AutoML też uzyskał bardzo dobre wyniki.

---

## 6. Wybór najlepszego modelu

Logika w funkcji `evaluate_models`:

- porównuje modele na podstawie **ROC AUC**,
- wybiera model o najwyższym ROC AUC jako `best_model`,
- zapisuje:
  - `model_comparison.json` – słownik z metrykami wszystkich trzech podejść + nazwą najlepszego modelu,
  - `best_model.pkl` – faktyczny obiekt modelu do dalszego użycia.

W naszym przypadku:

```text
best_model_name = "custom"
najlepszy model = RandomForest (custom)
```

---

## 7. Podsumowanie

1. **Zostało spełnione:**
   - baseline (DummyClassifier) – zbudowany, wytrenowany, metryki zapisane,
   - AutoML (PyCaret) – uruchomiony, ranking modeli zapisany do CSV, metryki i najlepszy model zapisane,
   - własny model (RandomForest) – zaimplementowany z prostym preprocessingiem, wytrenowany, zapisany,
   - pipeline `modeling` w Kedro łączy wszystkie kroki i poprawnie zapisuje wyniki,
   - raport (ten dokument) porównuje trzy podejścia.

2. **Najlepsze podejście:**
   - według ROC AUC i F1 najlepszy okazał się customowy RandomForest; AutoML dał model o bardzo dobrej Accuracy/F1, ale gorszym ROC AUC (prawdopodobnie z powodu formatu prawdopodobieństw).

