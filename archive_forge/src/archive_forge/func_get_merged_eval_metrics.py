import logging
import os
import shutil
import subprocess
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
from mlflow.exceptions import BAD_REQUEST, INVALID_PARAMETER_VALUE, MlflowException
from mlflow.recipes.cards import pandas_renderer
from mlflow.utils.databricks_utils import (
from mlflow.utils.os import is_windows
def get_merged_eval_metrics(eval_metrics: Dict[str, Dict], ordered_metric_names: Optional[List[str]]=None):
    """
    Returns a merged Pandas DataFrame from a map of dataset to evaluation metrics.
    Optionally, the rows in the DataFrame are ordered by input ordered metric names.

    Args:
        eval_metrics: Dict maps from dataset name to a Dict of evaluation metrics, which itself
            is a map from metric name to metric value.
        ordered_metric_names: List containing metric names. The ordering of the output is
            determined by this list, if provided.

    Returns:
        Pandas DataFrame containing evaluation metrics. The DataFrame is indexed by metric
        name. Columns are dataset names.
    """
    from pandas import DataFrame
    merged_metrics = {}
    for src, metrics in eval_metrics.items():
        if src not in merged_metrics:
            merged_metrics[src] = {}
        merged_metrics[src].update(metrics)
    if ordered_metric_names is None:
        ordered_metric_names = []
    metric_names = set()
    for val in merged_metrics.values():
        metric_names.update(val.keys())
    missing_metrics = set(ordered_metric_names) - metric_names
    if len(missing_metrics) > 0:
        _logger.warning('Input metric names %s not found in eval metrics: %s', missing_metrics, metric_names)
        ordered_metric_names = [name for name in ordered_metric_names if name not in missing_metrics]
    ordered_metric_names.extend(sorted(metric_names - set(ordered_metric_names)))
    return DataFrame.from_dict(merged_metrics).reindex(ordered_metric_names)