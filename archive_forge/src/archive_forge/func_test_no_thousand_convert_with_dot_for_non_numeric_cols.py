from __future__ import annotations
import csv
from io import (
from typing import TYPE_CHECKING
import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', [{'a': object}, {'a': str, 'b': np.int64, 'c': np.int64}])
def test_no_thousand_convert_with_dot_for_non_numeric_cols(python_parser_only, dtype):
    parser = python_parser_only
    data = 'a;b;c\n0000.7995;16.000;0\n3.03.001.00514;0;4.000\n4923.600.041;23.000;131'
    result = parser.read_csv(StringIO(data), sep=';', dtype=dtype, thousands='.')
    expected = DataFrame({'a': ['0000.7995', '3.03.001.00514', '4923.600.041'], 'b': [16000, 0, 23000], 'c': [0, 4000, 131]})
    tm.assert_frame_equal(result, expected)