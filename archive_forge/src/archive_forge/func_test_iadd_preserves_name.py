import datetime
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_iadd_preserves_name(self):
    ser = Series([1, 2, 3])
    ser.index.name = 'foo'
    ser.index += 1
    assert ser.index.name == 'foo'
    ser.index -= 1
    assert ser.index.name == 'foo'