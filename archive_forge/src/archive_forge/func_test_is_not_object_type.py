import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.types import (
def test_is_not_object_type(self, dtype):
    assert not is_object_dtype(dtype)