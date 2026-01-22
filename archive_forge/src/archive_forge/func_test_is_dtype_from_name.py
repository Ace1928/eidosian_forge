import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.types import (
def test_is_dtype_from_name(self, dtype):
    result = type(dtype).is_dtype(dtype.name)
    assert result is True