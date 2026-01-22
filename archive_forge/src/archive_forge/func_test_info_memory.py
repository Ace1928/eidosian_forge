from io import StringIO
from string import ascii_uppercase as uppercase
import textwrap
import numpy as np
import pytest
from pandas.compat import PYPY
from pandas import (
def test_info_memory():
    s = Series([1, 2], dtype='i8')
    buf = StringIO()
    s.info(buf=buf)
    result = buf.getvalue()
    memory_bytes = float(s.memory_usage())
    expected = textwrap.dedent(f"    <class 'pandas.core.series.Series'>\n    RangeIndex: 2 entries, 0 to 1\n    Series name: None\n    Non-Null Count  Dtype\n    --------------  -----\n    2 non-null      int64\n    dtypes: int64(1)\n    memory usage: {memory_bytes} bytes\n    ")
    assert result == expected