import datetime
import functools
from functools import partial
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_func_duplicates_raises():
    msg = 'Function names'
    df = DataFrame({'A': [0, 0, 1, 1], 'B': [1, 2, 3, 4]})
    with pytest.raises(SpecificationError, match=msg):
        df.groupby('A').agg(['min', 'min'])