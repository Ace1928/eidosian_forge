import importlib
import logging
from typing import TYPE_CHECKING, Any, Dict, Tuple
import pandas as pd
import mlflow
from mlflow import MlflowException
from mlflow.models import EvaluationMetric
from mlflow.models.evaluation.default_evaluator import (
from mlflow.recipes.utils.metrics import RecipeMetric, _load_custom_metrics
def calc_metric(X, y, estimator) -> Dict[str, float]:
    y_pred = estimator.predict(X)
    builtin_metrics = _get_regressor_metrics(y, y_pred, sample_weights=None) if task == 'regression' else _get_binary_classifier_metrics(y_true=y, y_pred=y_pred)
    res_df = pd.DataFrame()
    res_df['prediction'] = y_pred
    res_df['target'] = y if task == 'classification' else y.values
    return eval_metric.eval_fn(res_df, builtin_metrics)