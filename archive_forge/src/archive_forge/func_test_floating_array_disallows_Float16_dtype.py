import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import FloatingArray
from pandas.core.arrays.floating import (
def test_floating_array_disallows_Float16_dtype(request):
    with pytest.raises(TypeError, match="data type 'Float16' not understood"):
        pd.array([1.0, 2.0], dtype='Float16')