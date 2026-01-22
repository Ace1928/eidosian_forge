import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import FloatingArray
from pandas.core.arrays.floating import (
def test_uses_pandas_na():
    a = pd.array([1, None], dtype=Float64Dtype())
    assert a[1] is pd.NA