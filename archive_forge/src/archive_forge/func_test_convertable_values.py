from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
def test_convertable_values(self):
    with pytest.raises(TypeError, match="Could not convert string '1' to numeric"):
        nanops._ensure_numeric('1')
    with pytest.raises(TypeError, match="Could not convert string '1.1' to numeric"):
        nanops._ensure_numeric('1.1')
    with pytest.raises(TypeError, match="Could not convert string '1\\+1j' to numeric"):
        nanops._ensure_numeric('1+1j')