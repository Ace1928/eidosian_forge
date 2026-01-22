import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.arrays import BooleanArray
from pandas.core.arrays.boolean import coerce_to_array
def test_to_boolean_array_from_strings():
    result = BooleanArray._from_sequence_of_strings(np.array(['True', 'False', '1', '1.0', '0', '0.0', np.nan], dtype=object), dtype='boolean')
    expected = BooleanArray(np.array([True, False, True, True, False, False, False]), np.array([False, False, False, False, False, False, True]))
    tm.assert_extension_array_equal(result, expected)