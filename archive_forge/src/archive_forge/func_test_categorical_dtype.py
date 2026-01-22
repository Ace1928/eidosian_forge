from datetime import datetime
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas.compat import (
from pandas.compat.numpy import np_version_lt1p23
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.core.interchange.column import PandasColumn
from pandas.core.interchange.dataframe_protocol import (
from pandas.core.interchange.from_dataframe import from_dataframe
from pandas.core.interchange.utils import ArrowCTypes
@pytest.mark.parametrize('data', [('ordered', True), ('unordered', False)])
def test_categorical_dtype(data, data_categorical):
    df = pd.DataFrame({'A': data_categorical[data[0]]})
    col = df.__dataframe__().get_column_by_name('A')
    assert col.dtype[0] == DtypeKind.CATEGORICAL
    assert col.null_count == 0
    assert col.describe_null == (ColumnNullType.USE_SENTINEL, -1)
    assert col.num_chunks() == 1
    desc_cat = col.describe_categorical
    assert desc_cat['is_ordered'] == data[1]
    assert desc_cat['is_dictionary'] is True
    assert isinstance(desc_cat['categories'], PandasColumn)
    tm.assert_series_equal(desc_cat['categories']._col, pd.Series(['a', 'd', 'e', 's', 't']))
    tm.assert_frame_equal(df, from_dataframe(df.__dataframe__()))