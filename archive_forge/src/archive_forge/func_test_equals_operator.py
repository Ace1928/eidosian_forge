import numpy as np
import pytest
from pandas.core.dtypes.common import is_any_real_numeric_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_equals_operator(idx):
    assert (idx == idx).all()