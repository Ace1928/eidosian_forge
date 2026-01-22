import io
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util import _test_decorators as td
@td.skip_if_installed('fsspec')
def test_not_present_exception():
    msg = "Missing optional dependency 'fsspec'|fsspec library is required"
    with pytest.raises(ImportError, match=msg):
        read_csv('memory://test/test.csv')