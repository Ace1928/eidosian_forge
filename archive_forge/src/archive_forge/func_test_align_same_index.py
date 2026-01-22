from datetime import timezone
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_align_same_index(datetime_series, using_copy_on_write):
    a, b = datetime_series.align(datetime_series, copy=False)
    if not using_copy_on_write:
        assert a.index is datetime_series.index
        assert b.index is datetime_series.index
    else:
        assert a.index.is_(datetime_series.index)
        assert b.index.is_(datetime_series.index)
    a, b = datetime_series.align(datetime_series, copy=True)
    assert a.index is not datetime_series.index
    assert b.index is not datetime_series.index
    assert a.index.is_(datetime_series.index)
    assert b.index.is_(datetime_series.index)