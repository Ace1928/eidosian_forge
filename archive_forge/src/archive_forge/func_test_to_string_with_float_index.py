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
def test_to_string_with_float_index(self):
    index = Index([1.5, 2, 3, 4, 5])
    df = DataFrame(np.arange(5), index=index)
    result = df.to_string()
    expected = '     0\n1.5  0\n2.0  1\n3.0  2\n4.0  3\n5.0  4'
    assert result == expected