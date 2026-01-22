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
def test_getitem_preserve_name(datetime_series):
    result = datetime_series[datetime_series > 0]
    assert result.name == datetime_series.name
    msg = 'Series.__getitem__ treating keys as positions is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = datetime_series[[0, 2, 4]]
    assert result.name == datetime_series.name
    result = datetime_series[5:10]
    assert result.name == datetime_series.name