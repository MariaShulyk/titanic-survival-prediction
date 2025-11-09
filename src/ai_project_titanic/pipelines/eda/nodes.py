from __future__ import annotations
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


def basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    log.info("DF shape: %s rows, %s cols", *df.shape)
    # desc = df.describe(include="all", datetime_is_numeric=True).T
    desc = df.describe(include="all").transpose()
    dtypes = df.dtypes.astype(str).rename("dtype")
    out = desc.merge(dtypes.to_frame(), left_index=True, right_index=True, how="left")
    return out.reset_index(names="column")


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum().to_frame("missing_count")
    miss["missing_pct"] = (miss["missing_count"] / len(df) * 100).round(2)
    return miss.reset_index(names="column").sort_values("missing_pct", ascending=False)


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number])

    if num.empty:
        log.warning("No numeric columns for correlations.")
        return pd.DataFrame()

    return num.corr(numeric_only=True)


def detect_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number])
    rows = []

    for col in num.columns:
        q1, q3 = num[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        cnt = ((num[col] < lo) | (num[col] > hi)).sum()
        rows.append({"column": col, "iqr_low": lo, "iqr_high": hi, "outliers_count": int(cnt)})
    return pd.DataFrame(rows).sort_values("outliers_count", ascending=False)


def _fig():
    return plt.figure()


def plot_histograms(df: pd.DataFrame) -> plt.Figure:
    num = df.select_dtypes(include=[np.number])
    cols = list(num.columns)
    if not cols:
        return _fig()
    n = len(cols)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, cols):
        ax.hist(num[col].dropna(), bins=30)
        ax.set_title(f"Histogram: {col}")

    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_boxplots(df: pd.DataFrame) -> plt.Figure:
    num = df.select_dtypes(include=[np.number])
    cols = list(num.columns)
    if not cols:
        return _fig()
    n = len(cols)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, cols):
        ax.boxplot(num[col].dropna())
        ax.set_title(f"Boxplot: {col}")

    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_countplots(df: pd.DataFrame) -> plt.Figure:
    cat = df.select_dtypes(exclude=[np.number])
    cols = list(cat.columns)
    if not cols:
        return _fig()
    n = len(cols)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, cols):
        vc = cat[col].astype(str).value_counts().head(20)
        ax.bar(vc.index, vc.values)
        ax.set_title(f"Count: {col}")
        ax.tick_params(axis="x", labelrotation=90)

    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_corr_heatmap(df: pd.DataFrame) -> plt.Figure:
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return _fig()
    corr = num.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(max(6, 0.5*len(corr)), max(4, 0.5*len(corr))))
    cax = ax.imshow(corr, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.index)
    fig.colorbar(cax)
    ax.set_title("Correlation heatmap")
    fig.tight_layout()
    return fig


def plot_pairplot(df: pd.DataFrame) -> plt.Figure:
    num = df.select_dtypes(include=[np.number]).dropna()
    cols = list(num.columns)
    if len(cols) < 2:
        return _fig()
    cols = cols[:6]
    n = len(cols)
    fig, axes = plt.subplots(n, n, figsize=(3*n, 3*n))

    for i, x in enumerate(cols):
        for j, y in enumerate(cols):
            ax = axes[i, j]
            if i == j:
                ax.hist(num[x], bins=30)
            else:
                ax.scatter(num[y], num[x], s=5, alpha=0.5)
            if i == n-1:
                ax.set_xlabel(y)
            else:
                ax.set_xticks([])
            if j == 0:
                ax.set_ylabel(x)
            else:
                ax.set_yticks([])

    fig.suptitle("Pairplot (basic)")
    fig.tight_layout()
    return fig
