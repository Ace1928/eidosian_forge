from typing import List, Tuple
import numpy as np
import pandas as pd
from ray.data import Dataset
from ray.data.aggregate import AbsMax, Max, Mean, Min, Std
from ray.data.preprocessor import Preprocessor
from ray.util.annotations import PublicAPI
def column_min_max_scaler(s: pd.Series):
    s_min = self.stats_[f'min({s.name})']
    s_max = self.stats_[f'max({s.name})']
    diff = s_max - s_min
    if diff == 0:
        diff = 1
    return (s - s_min) / diff