import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.types import (
def test_construct_from_string_wrong_type_raises(self, dtype):
    with pytest.raises(TypeError, match="'construct_from_string' expects a string, got <class 'int'>"):
        type(dtype).construct_from_string(0)