from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
def get_categorical_invalid_expected():
    lev = Categorical([0], dtype=values.dtype)
    if len(keys) != 1:
        idx = MultiIndex.from_product([lev, lev], names=keys)
    else:
        idx = Index(lev, name=keys[0])
    if using_infer_string:
        columns = Index([], dtype='string[pyarrow_numpy]')
    else:
        columns = []
    expected = DataFrame([], columns=columns, index=idx)
    return expected