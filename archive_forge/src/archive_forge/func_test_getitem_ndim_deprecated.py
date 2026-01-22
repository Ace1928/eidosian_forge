from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexing import IndexingError
from pandas.tseries.offsets import BDay
@pytest.mark.parametrize('series', [Series([0, 1]), Series(date_range('2012-01-01', periods=2)), Series(date_range('2012-01-01', periods=2, tz='CET'))])
def test_getitem_ndim_deprecated(series):
    with pytest.raises(ValueError, match='Multi-dimensional indexing'):
        series[:, None]