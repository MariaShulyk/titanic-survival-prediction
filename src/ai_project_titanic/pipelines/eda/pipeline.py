from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    basic_stats, missing_summary, correlations, detect_outliers_iqr,
    plot_histograms, plot_boxplots, plot_countplots, plot_corr_heatmap, plot_pairplot
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(basic_stats, "raw_titanic", "eda_basic_stats", name="eda_basic_stats"),
        node(missing_summary, "raw_titanic", "eda_missing_summary", name="eda_missing_summary"),
        node(correlations, "raw_titanic", "eda_correlations", name="eda_correlations"),
        node(detect_outliers_iqr, "raw_titanic", "eda_outliers_iqr", name="eda_outliers_iqr"),
        node(plot_histograms, "raw_titanic", "eda_plots_hist", name="eda_plot_histograms"),
        node(plot_boxplots, "raw_titanic", "eda_plots_box", name="eda_plot_boxplots"),
        node(plot_countplots, "raw_titanic", "eda_plots_count", name="eda_plot_countplots"),
        node(plot_corr_heatmap, "raw_titanic", "eda_plots_corr_heatmap", name="eda_plot_corr_heatmap"),
        node(plot_pairplot, "raw_titanic", "eda_plots_pairplot", name="eda_plot_pairplot"),
    ])
