import numpy as np
import pandas as pd
from enum import Enum
from .. import CatBoostError
from ..core import metric_description_or_str_to_str
from ..utils import compute_wx_test
def get_fold_ids(self):
    """

        :return: Folds ids which we used for computing this evaluation result
        """
    return self._case_results[self._baseline_case].get_fold_ids()