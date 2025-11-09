# EDA Report – Titanic

## 1. Opis danych

Dane pochodzą z pliku `titanic.csv`.  
Zestaw zawiera 891 wierszy oraz 12 kolumn opisujących pasażerów statku Titanic.  
Dane obejmują zarówno cechy demograficzne, jak i informacje o bilecie, kabinie oraz ostatecznym wyniku podróży (przeżycie).

Kolumny w zbiorze:
- PassengerId – unikalny identyfikator pasażera.
- Survived – zmienna wyjściowa (1 = przeżył, 0 = nie przeżył).
- Pclass – klasa biletu (1, 2, 3), silnie związana z poziomem luksusu i dostępem do łodzi ratunkowych.
- Name – pełne imię i nazwisko pasażera (zawiera tytuły).
- Sex – płeć.
- Age – wiek (ciągła zmienna, zawiera braki).
- SibSp – liczba rodzeństwa/małżonków na pokładzie.
- Parch – liczba rodziców/dzieci na pokładzie.
- Ticket – numer biletu.
- Fare – cena biletu, zmienna ciągła, mocno skośna.
- Cabin – numer kabiny (bardzo dużo braków).
- Embarked – port wejścia na statek (C, Q, S).

## 2. Główne obserwacje

- Najwięcej braków danych występuje w `Cabin` (około 77% braków). Zmienna ta jest prawdopodobnie zbyt niekompletna, aby używać jej bez przekształceń.
- Kolumna `Age` ma około 19% braków, co wymaga imputacji.
- `Embarked` ma pojedyncze braki (0.22%), łatwe do uzupełnienia.
- W zmiennych `Fare`, `SibSp` oraz `Parch` obecne są wartości odstające. Dotyczy to szczególnie wysokich wartości `Fare`.
- Rozkład `Age` jest lekko prawostronnie skośny, a `Fare` jest silnie skośny, co sugeruje transformację (np. logarytmiczną).
- Z analizy korelacji wynika, że największy związek z `Survived` mają zmienne:
  - `Sex` (kobiety przeżywały znacznie częściej),
  - `Pclass` (pasażerowie 1 klasy mieli najwyższe szanse),
  - `Fare` (droższe bilety częściej wiązały się z wyższą klasą).
- `SibSp` i `Parch` tworzą wzorce związane z wielkością rodziny; osoby samotne miały niższy wskaźnik przeżycia.
- Zmienna `Ticket` jest niejednoznaczna i wymaga dalszej obróbki lub zakodowania grupowego.
- `Cabin` może zawierać istotne informacje o pokładzie, jeśli wyodrębnimy literę (np. C, B, E), ale w wielu przypadkach nie jest znana.

## 3. Wygenerowane wyniki (z pipeline EDA)

- **eda_basic_stats** zawiera pełne statystyki opisowe, typy danych oraz zróżnicowanie kolumn.  
  Wskazuje m.in. na duży rozrzut wartości w `Fare` oraz brak standaryzacji wielu cech.
- **eda_missing_summary** pokazuje procent braków danych w każdej kolumnie, potwierdzając największe braki w `Cabin` i umiarkowane w `Age`.
- **eda_correlations** prezentuje macierz korelacji między zmiennymi numerycznymi.  
  Najważniejsze obserwacje:
  - `Pclass` i `Fare` są skorelowane (negatywnie).
  - `Age` ma niskie korelacje z większością cech.
  - `Survived` koreluje najmocniej z `Fare` oraz `Pclass`.
- **eda_outliers_iqr** identyfikuje wartości odstające metodą IQR.  
  Najwięcej outlierów znajduje się w `Fare`, `Parch` i `SibSp`, co sugeruje potrzebę dalszej analizy lub transformacji.
- Wygenerowane wykresy (histogramy, boxploty, wykresy liczności, mapa korelacji, pairplot) potwierdzają zależności i skośność wielu zmiennych.

## 4. Proponowane dalsze kroki

### Czyszczenie
- Imputacja `Age` medianą lub modelowo (regresja oparta o Pclass, Sex i FamilySize).
- Uzupełnienie `Embarked` wartością modu (S).
- Redukcja `Cabin` do zmiennej:
  - binarnej (CabinKnown),
  - lub literowej (deck).
- Sprawdzenie sensowności zmiennej `Ticket` (możliwe grupowanie po prefiksach).

### Standaryzacja
- Skalowanie cech `Age` i `Fare` przy użyciu StandardScaler lub RobustScaler.
- Transformacja logarytmiczna dla `Fare` w celu zmniejszenia skośności.

### Feature engineering
- Wyodrębnienie tytułów z kolumny `Name` (Mr, Mrs, Miss, Master itd.) jako silnie predykcyjnej cechy.
- Stworzenie zmiennej `FamilySize = SibSp + Parch + 1`.
- Dodanie cechy `IsAlone` (1 jeśli FamilySize == 1).
- Wyodrębnienie litery pokładu z `Cabin` (np. C, B, E) i potraktowanie jej jako cechy kategorialnej.
- Grupowanie numerów biletów lub ekstrakcja prefiksów z `Ticket`.

### Pipeline modelowania
- Stworzenie preprocessing pipeline z:
  - imputacją braków,
  - one-hot encodingiem zmiennych kategorycznych,
  - skalowaniem zmiennych numerycznych,
  - ewentualną transformacją logarytmiczną.
- Podział na zbiór treningowy i testowy.
- Przygotowanie kilku modeli klasyfikacyjnych:
  - Logistic Regression,
  - Random Forest,
  - Gradient Boosting,
  - XGBoost (opcjonalnie).

