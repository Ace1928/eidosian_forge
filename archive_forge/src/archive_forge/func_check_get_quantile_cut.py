import json
from functools import partial, update_wrapper
from typing import Any, Dict, List
import numpy as np
import xgboost as xgb
import xgboost.testing as tm
def check_get_quantile_cut(tree_method: str) -> None:
    """Check the quantile cut getter."""
    use_cupy = tree_method == 'gpu_hist'
    check_get_quantile_cut_device(tree_method, False)
    if use_cupy:
        check_get_quantile_cut_device(tree_method, True)