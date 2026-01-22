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
def truncate_pandas_data_profile(title: str, data_frame) -> str:
    """
    Returns a data profiling string over input data frame.

    Args:
        title: The title of the data profile.
        data_frame: Contains data to be profiled.

    Returns:
        A data profiling string such as Pandas profiling ProfileReport.
    """
    if len(data_frame) == 0:
        return (title, data_frame)
    max_cells = min(data_frame.size, _MAX_PROFILE_CELL_SIZE)
    max_cols = min(data_frame.columns.size, _MAX_PROFILE_COL_SIZE)
    max_rows = min(max(max_cells // max_cols, 1), len(data_frame), _MAX_PROFILE_ROW_SIZE)
    truncated_df = data_frame.drop(columns=data_frame.columns[max_cols:]).sample(n=max_rows, ignore_index=True, random_state=42)
    if max_cells == _MAX_PROFILE_CELL_SIZE or max_cols == _MAX_PROFILE_COL_SIZE or max_rows == _MAX_PROFILE_ROW_SIZE:
        _logger.info('Truncating the data frame for %s to %d cells, %d columns and %d rows', title, max_cells, max_cols, max_rows)
    return (title, truncated_df)