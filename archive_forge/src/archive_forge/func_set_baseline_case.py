import numpy as np
import pandas as pd
from enum import Enum
from .. import CatBoostError
from ..core import metric_description_or_str_to_str
from ..utils import compute_wx_test
def set_baseline_case(self, case):
    """
            Could be used to change baseline cases for already computed results
        """
    for metric, metric_result in self._results.items():
        metric_result.change_baseline_case(case)