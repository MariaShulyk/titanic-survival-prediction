import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Fare"] = pd.to_numeric(df["Fare"], errors="coerce")

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    df["CabinKnown"] = df["Cabin"].notna().astype(int)
    df.drop(columns=["Cabin"], inplace=True)

    def remove_outliers_iqr(series: pd.Series) -> pd.Series:
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return series.clip(lower, upper)

    df["Fare"] = remove_outliers_iqr(df["Fare"])
    df["Age"] = remove_outliers_iqr(df["Age"])

    return df



def scale_data(df: pd.DataFrame) -> pd.DataFrame:
    TARGET_COL = "Survived"
    df_scaled = df.copy()

    numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()

    if TARGET_COL in numeric_cols:
        numeric_cols.remove(TARGET_COL)

    scaler = StandardScaler()
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])

    return df_scaled


def split_data(df: pd.DataFrame):
    train, temp = train_test_split(df, test_size=0.30, random_state=42)
    val, test = train_test_split(temp, test_size=0.50, random_state=42)

    return train, val, test
