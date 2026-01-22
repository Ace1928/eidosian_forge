from datetime import (
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.frequencies import to_offset
def test_dti_contains_with_duplicates(self):
    d = datetime(2011, 12, 5, 20, 30)
    ix = DatetimeIndex([d, d])
    assert d in ix