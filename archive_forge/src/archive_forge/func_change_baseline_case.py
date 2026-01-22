import numpy as np
import pandas as pd
from enum import Enum
from .. import CatBoostError
from ..core import metric_description_or_str_to_str
from ..utils import compute_wx_test
def change_baseline_case(self, case):
    """

        :param case: new baseline case
        :return:
        """
    if case not in self._case_results:
        raise CatBoostError("Case {} is unknown. Can't use it as baseline".format(case))
    self._baseline_case = case