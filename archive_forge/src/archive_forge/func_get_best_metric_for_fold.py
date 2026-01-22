import numpy as np
import pandas as pd
from enum import Enum
from .. import CatBoostError
from ..core import metric_description_or_str_to_str
from ..utils import compute_wx_test
def get_best_metric_for_fold(self, fold):
    """

        :param fold: id of fold to get result
        :return: best metric value, best metric iteration
        """
    return (self._fold_metric[fold], self._fold_metric_iteration[fold])