from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_contains_with_missing_value(self):
    idx = MultiIndex.from_arrays([[1, np.nan, 2]])
    assert np.nan in idx
    idx = MultiIndex.from_arrays([[1, 2], [np.nan, 3]])
    assert np.nan not in idx
    assert (1, np.nan) in idx