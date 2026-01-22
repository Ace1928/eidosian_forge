import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.types import (
def test_is_not_string_type(self, dtype):
    assert not is_string_dtype(dtype)