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
def test_empty_string_column():
    df = pd.DataFrame({'a': []}, dtype=str)
    df2 = df.__dataframe__()
    result = pd.api.interchange.from_dataframe(df2)
    tm.assert_frame_equal(df, result)