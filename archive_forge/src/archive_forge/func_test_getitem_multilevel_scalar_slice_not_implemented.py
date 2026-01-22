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
def test_getitem_multilevel_scalar_slice_not_implemented(multiindex_year_month_day_dataframe_random_data):
    df = multiindex_year_month_day_dataframe_random_data
    ser = df['A']
    msg = '\\(2000, slice\\(3, 4, None\\)\\)'
    with pytest.raises(TypeError, match=msg):
        ser[2000, 3:4]