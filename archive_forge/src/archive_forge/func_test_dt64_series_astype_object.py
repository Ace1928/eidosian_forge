from datetime import (
from importlib import reload
import string
import sys
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_dt64_series_astype_object(self):
    dt64ser = Series(date_range('20130101', periods=3))
    result = dt64ser.astype(object)
    assert isinstance(result.iloc[0], datetime)
    assert result.dtype == np.object_