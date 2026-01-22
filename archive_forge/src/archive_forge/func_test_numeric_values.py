from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
def test_numeric_values(self):
    assert nanops._ensure_numeric(1) == 1
    assert nanops._ensure_numeric(1.1) == 1.1
    assert nanops._ensure_numeric(1 + 2j) == 1 + 2j