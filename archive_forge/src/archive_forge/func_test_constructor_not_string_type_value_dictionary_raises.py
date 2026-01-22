import pickle
import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.string_ import (
from pandas.core.arrays.string_arrow import (
@pytest.mark.parametrize('chunked', [True, False])
def test_constructor_not_string_type_value_dictionary_raises(chunked):
    pa = pytest.importorskip('pyarrow')
    arr = pa.array([1, 2, 3], pa.dictionary(pa.int32(), pa.int32()))
    if chunked:
        arr = pa.chunked_array(arr)
    msg = re.escape('ArrowStringArray requires a PyArrow (chunked) array of large_string type')
    with pytest.raises(ValueError, match=msg):
        ArrowStringArray(arr)