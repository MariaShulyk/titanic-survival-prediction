# Preprocessing Report – Titanic

## 1. Cel preprocessing’u

Celem preprocessing pipeline’u było przygotowanie danych z pliku `Titanic-Dataset.csv` do dalszego modelowania.  
Zadaniem było:

- oczyszczenie danych,
- uzupełnienie braków,
- usunięcie wartości odstających,
- standaryzacja zmiennych numerycznych,
- podział danych na zestawy treningowe, walidacyjne i testowe,
- zbudowanie kompletnego pipeline’u w Kedro.

Pipeline jest uruchamiany poleceniem:

```
kedro run --pipeline=preprocessing
```

---

## 2. Dane wejściowe

Dane wejściowe znajdują się w pliku:

- **data/01_raw/Titanic-Dataset.csv**

Zbiór zawiera 891 wierszy oraz 12 kolumn, w tym cechy:

- PassengerId, Survived, Pclass
- Name, Sex
- Age (braki danych)
- SibSp, Parch
- Ticket
- Fare (wartości odstające)
- Cabin (bardzo dużo braków)
- Embarked (pojedyncze braki)

---

## 3. Etap 1 – Czyszczenie danych

Węzeł: **clean_data_node**  
Plik wyjściowy: **data/02_intermediate/clean_data.csv**

### 3.1. Uzupełnianie braków

- `Age` – uzupełniono medianą.  
- `Embarked` – uzupełniono najczęściej występującą wartością (modą).  
- `Cabin` – z 77% braków → zmieniono na cechę binarną:

  - `CabinKnown = 1` jeśli kabina jest znana  
  - `CabinKnown = 0` jeśli brak

  Oryginalną kolumnę `Cabin` usunięto.

### 3.2. Obsługa typów danych

- `Age` i `Fare` zamieniono na `float`.  
- Pozostałe kolumny pozostawiono bez zmian.

### 3.3. Usuwanie wartości odstających (IQR)

Dla kolumn `Age` i `Fare` zastosowano clipping metodą IQR:

- obliczenie Q1, Q3, IQR,  
- wartości spoza zakresu `[Q1 − 1.5*IQR, Q3 + 1.5*IQR]` zostały przycięte.

Metoda zmniejsza wpływ wartości odstających bez usuwania wierszy.

---

## 4. Etap 2 – Standaryzacja danych

Węzeł: **scale_data_node**  
Plik wyjściowy: **data/03_primary/scaled_data.csv**

Standaryzacja została wykonana na kolumnach numerycznych:

- PassengerId  
- Survived  
- Pclass  
- Age  
- SibSp  
- Parch  
- Fare  
- CabinKnown  

Zastosowano **StandardScaler**, który:

- ustawia średnią = 0  
- ustawia odchylenie standardowe = 1  

Testy jednostkowe potwierdzają, że średnia ≈ 0 i std ≈ 1 (ddof=0, zgodne z implementacją Scikit-Learn).

---

## 5. Etap 3 – Podział danych

Węzeł: **split_data_node**  
Pliki wyjściowe:

- **data/05_model_input/train.csv**
- **data/05_model_input/val.csv**
- **data/05_model_input/test.csv**

Zastosowano podział:

- **70%** – zbiór treningowy  
- **15%** – zbiór walidacyjny  
- **15%** – zbiór testowy  

Wykorzystano `train_test_split` z `random_state=42`.

---

## 6. Implementacja pipeline’u

Struktura modułu:

```
pipelines/
└── preprocessing/
    ├── nodes.py
    ├── pipeline.py
    └── __init__.py
```

Pipeline:

```
Pipeline([
    node(clean_data, "raw_titanic", "clean_data"),
    node(scale_data, "clean_data", "scaled_data"),
    node(split_data, "scaled_data", ["train_data", "val_data", "test_data"]),
])
```

Rejestracja w pipeline_registry.py:

```
from ai_project_titanic.pipelines import preprocessing as preprocessing_pipeline

def register_pipelines():
    preprocessing = preprocessing_pipeline.create_pipeline()
    return {
        "preprocessing": preprocessing
    }
```

---

## 7. Testy jednostkowe

Plik testowy:  
`tests/pipelines/preprocessing/test_nodes.py`

Przetestowano:

- **clean_data** – poprawna imputacja, usunięcie `Cabin`, dodanie `CabinKnown`.  
- **scale_data** – poprawna standaryzacja (`mean≈0`, `std≈1`).  
- **split_data** – poprawny podział w proporcjach 70/15/15.

Wynik:

```
3 passed, 0 failed
```

---

## 8. Wyniki końcowe

Pipeline generuje następujące dane przetworzone:

- **data/02_intermediate/clean_data.csv** – dane po czyszczeniu  
- **data/03_primary/scaled_data.csv** – dane po standaryzacji  
- **data/05_model_input/train.csv** – zbiór treningowy  
- **data/05_model_input/val.csv** – zbiór walidacyjny  
- **data/05_model_input/test.csv** – zbiór testowy  

Pipeline działa w pełni automatycznie i poprawnie.

---

## 9. Podsumowanie

Zrealizowano kompletny preprocessing:

- imputacja braków,  
- redukcja kolumny `Cabin`,  
- usunięcie outlierów metodą IQR,  
- standaryzacja danych,  
- podział zbiorów,  
- implementacja pipeline’u Kedro,  
- pełne testy jednostkowe.  

Dane są gotowe do dalszego modelowania.
