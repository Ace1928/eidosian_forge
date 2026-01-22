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
def validate_classification_config(task: str, positive_class: str, input_df: pd.DataFrame, target_col: str):
    """
    Args:
        task:
        positive_class:
        input_df:
        target_col:
    """
    if task == 'classification':
        classes = np.unique(input_df[target_col])
        num_classes = len(classes)
        if num_classes <= 1:
            raise MlflowException(f'Classification tasks require at least two tasks. Your dataset contains {num_classes}.')
        elif positive_class is None and num_classes == 2:
            raise MlflowException('`positive_class` must be specified for classification/v1 recipes.', error_code=INVALID_PARAMETER_VALUE)