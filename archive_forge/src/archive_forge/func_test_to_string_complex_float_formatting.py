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
def test_to_string_complex_float_formatting(self):
    with option_context('display.precision', 5):
        df = DataFrame({'x': [0.4467846931321966 + 0.0715185102060818j, 0.2739442392974528 + 0.23515228785438969j, 0.26974928742135185 + 0.3250604054898979j, -1j]})
        result = df.to_string()
        expected = '                  x\n0  0.44678+0.07152j\n1  0.27394+0.23515j\n2  0.26975+0.32506j\n3 -0.00000-1.00000j'
        assert result == expected