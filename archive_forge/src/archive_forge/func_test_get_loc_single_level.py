import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_get_loc_single_level(self, single_level_multiindex):
    single_level = single_level_multiindex
    s = Series(np.random.default_rng(2).standard_normal(len(single_level)), index=single_level)
    for k in single_level.values:
        s[k]