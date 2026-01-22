from itertools import product
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
def test_subset_overlaps_gb_key_raises():
    df = DataFrame({'c1': ['a', 'b', 'c'], 'c2': ['x', 'y', 'y']}, index=[0, 1, 1])
    msg = "Keys {'c1'} in subset cannot be in the groupby column keys."
    with pytest.raises(ValueError, match=msg):
        df.groupby('c1').value_counts(subset=['c1'])