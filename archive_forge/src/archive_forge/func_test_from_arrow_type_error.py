import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.arrow._arrow_utils import pyarrow_array_to_numpy_and_mask
def test_from_arrow_type_error(data):
    arr = pa.array(data).cast('string')
    with pytest.raises(TypeError, match=None):
        data.dtype.__from_arrow__(arr)