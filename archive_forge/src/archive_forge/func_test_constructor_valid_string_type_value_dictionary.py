import pickle
import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.string_ import (
from pandas.core.arrays.string_arrow import (
@pytest.mark.xfail(reason='dict conversion does not seem to be implemented for large string in arrow')
@pytest.mark.parametrize('chunked', [True, False])
def test_constructor_valid_string_type_value_dictionary(chunked):
    pa = pytest.importorskip('pyarrow')
    arr = pa.array(['1', '2', '3'], pa.large_string()).dictionary_encode()
    if chunked:
        arr = pa.chunked_array(arr)
    arr = ArrowStringArray(arr)
    assert pa.types.is_string(arr._pa_array.type.value_type)