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
def test_to_string_repr_unicode(self):
    buf = StringIO()
    unicode_values = ['σ'] * 10
    unicode_values = np.array(unicode_values, dtype=object)
    df = DataFrame({'unicode': unicode_values})
    df.to_string(col_space=10, buf=buf)
    repr(df)
    _stdin = sys.stdin
    try:
        sys.stdin = None
        repr(df)
    finally:
        sys.stdin = _stdin