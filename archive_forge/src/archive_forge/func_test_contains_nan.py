import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_contains_nan(self):
    ci = CategoricalIndex(list('aabbca') + [np.nan], categories=list('cabdef'))
    assert np.nan in ci