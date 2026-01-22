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
def test_large_string():
    pytest.importorskip('pyarrow')
    df = pd.DataFrame({'a': ['x']}, dtype='large_string[pyarrow]')
    result = pd.api.interchange.from_dataframe(df.__dataframe__())
    expected = pd.DataFrame({'a': ['x']}, dtype='object')
    tm.assert_frame_equal(result, expected)