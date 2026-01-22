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
def test_to_string_decimal(self):
    df = DataFrame({'A': [6.0, 3.1, 2.2]})
    expected = '     A\n0  6,0\n1  3,1\n2  2,2'
    assert df.to_string(decimal=',') == expected