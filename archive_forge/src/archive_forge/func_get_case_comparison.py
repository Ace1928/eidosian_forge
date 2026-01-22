import numpy as np
import pandas as pd
from enum import Enum
from .. import CatBoostError
from ..core import metric_description_or_str_to_str
from ..utils import compute_wx_test
def get_case_comparison(self, case, score_config=None):
    """
        Method to get human-friendly table with model comparisons.
        Same as get_baseline_comparison(), but with other non-baseline case specified as baseline

        :param case: use specified case as baseline
        :param score_config:
        :return: pandas DataFrame. Each row is related to one ExecutionCase.
        Each row describes how better (or worse) this case is compared to baseline.
        """
    self._change_score_config(score_config)
    if case not in self._case_comparisons:
        self._case_comparisons[case] = self._compute_case_result_table(case)
    return self._case_comparisons[case]