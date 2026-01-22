import numpy as np
import pandas as pd
from enum import Enum
from .. import CatBoostError
from ..core import metric_description_or_str_to_str
from ..utils import compute_wx_test
def get_eval_step(self):
    return self._case_results[self._baseline_case].get_eval_step()