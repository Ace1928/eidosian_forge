from io import StringIO
import re
from string import ascii_uppercase
import sys
import textwrap
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
@pytest.mark.single_cpu
def test_info_compute_numba():
    pytest.importorskip('numba')
    df = DataFrame([[1, 2], [3, 4]])
    with option_context('compute.use_numba', True):
        buf = StringIO()
        df.info(buf=buf)
        result = buf.getvalue()
    buf = StringIO()
    df.info(buf=buf)
    expected = buf.getvalue()
    assert result == expected