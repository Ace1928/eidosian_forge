from itertools import product
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
def rebuild_index(df):
    arr = list(map(df.index.get_level_values, range(df.index.nlevels)))
    df.index = MultiIndex.from_arrays(arr, names=df.index.names)
    return df