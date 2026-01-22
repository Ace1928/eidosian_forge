import collections
import operator
import sys
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.json.array import (
@pytest.mark.xfail(reason='collection as scalar, GH-33901')
def test_series_constructor_scalar_with_index(self, data, dtype):
    rec_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(100)
        super().test_series_constructor_scalar_with_index(data, dtype)
    finally:
        sys.setrecursionlimit(rec_limit)