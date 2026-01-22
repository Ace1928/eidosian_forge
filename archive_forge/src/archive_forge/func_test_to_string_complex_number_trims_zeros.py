from datetime import (
from io import StringIO
import re
import sys
from textwrap import dedent
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
import pandas._testing as tm
def test_to_string_complex_number_trims_zeros(self):
    ser = Series([1.0 + 1j, 1.0 + 1j, 1.05 + 1j])
    result = ser.to_string()
    expected = dedent('            0    1.00+1.00j\n            1    1.00+1.00j\n            2    1.05+1.00j')
    assert result == expected