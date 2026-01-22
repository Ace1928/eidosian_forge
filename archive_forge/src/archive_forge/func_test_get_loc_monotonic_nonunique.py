import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_get_loc_monotonic_nonunique(self):
    cidx = CategoricalIndex(list('abbc'))
    result = cidx.get_loc('b')
    expected = slice(1, 3, None)
    assert result == expected