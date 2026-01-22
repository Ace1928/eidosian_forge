import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
import pandas.core.reshape.tile as tmod
def test_cut_out_of_bounds():
    arr = np.random.default_rng(2).standard_normal(100)
    result = cut(arr, [-1, 0, 1])
    mask = isna(result)
    ex_mask = (arr < -1) | (arr > 1)
    tm.assert_numpy_array_equal(mask, ex_mask)