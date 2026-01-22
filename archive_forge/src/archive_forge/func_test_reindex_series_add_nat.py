import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_reindex_series_add_nat():
    rng = date_range('1/1/2000 00:00:00', periods=10, freq='10s')
    series = Series(rng)
    result = series.reindex(range(15))
    assert np.issubdtype(result.dtype, np.dtype('M8[ns]'))
    mask = result.isna()
    assert mask[-5:].all()
    assert not mask[:-5].any()