import builtins
from io import StringIO
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.tests.groupby import get_groupby_method_args
from pandas.util import _test_decorators as td
@pytest.fixture(params=[np.int32, np.int64, np.float32, np.float64, 'Int64', 'Float64'], ids=['np.int32', 'np.int64', 'np.float32', 'np.float64', 'Int64', 'Float64'])
def dtypes_for_minmax(request):
    """
    Fixture of dtypes with min and max values used for testing
    cummin and cummax
    """
    dtype = request.param
    np_type = dtype
    if dtype == 'Int64':
        np_type = np.int64
    elif dtype == 'Float64':
        np_type = np.float64
    min_val = np.iinfo(np_type).min if np.dtype(np_type).kind == 'i' else np.finfo(np_type).min
    max_val = np.iinfo(np_type).max if np.dtype(np_type).kind == 'i' else np.finfo(np_type).max
    return (dtype, min_val, max_val)