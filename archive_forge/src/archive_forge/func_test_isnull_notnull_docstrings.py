import sys
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import PYPY
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_isnull_notnull_docstrings():
    doc = pd.DataFrame.notnull.__doc__
    assert doc.startswith('\nDataFrame.notnull is an alias for DataFrame.notna.\n')
    doc = pd.DataFrame.isnull.__doc__
    assert doc.startswith('\nDataFrame.isnull is an alias for DataFrame.isna.\n')
    doc = Series.notnull.__doc__
    assert doc.startswith('\nSeries.notnull is an alias for Series.notna.\n')
    doc = Series.isnull.__doc__
    assert doc.startswith('\nSeries.isnull is an alias for Series.isna.\n')