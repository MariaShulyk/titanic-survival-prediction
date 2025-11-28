from __future__ import annotations
from kedro.pipeline import Pipeline
from ai_project_titanic.pipelines.eda import create_pipeline as create_eda_pipeline
from ai_project_titanic.pipelines import preprocessing as preprocessing_pipeline
from ai_project_titanic.pipelines import modeling as modeling_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    eda = create_eda_pipeline()
    preprocessing = preprocessing_pipeline.create_pipeline()
    modeling = modeling_pipeline.create_pipeline()

    return {
        "__default__": eda + preprocessing + modeling,
        "eda": eda,
        "preprocessing": preprocessing,
        "modeling": modeling,
    }
