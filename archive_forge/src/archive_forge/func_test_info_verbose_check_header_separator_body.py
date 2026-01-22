from io import StringIO
import re
from string import ascii_uppercase as uppercase
import sys
import textwrap
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
def test_info_verbose_check_header_separator_body():
    buf = StringIO()
    size = 1001
    start = 5
    frame = DataFrame(np.random.randn(3, size))
    frame.info(verbose=True, buf=buf)
    res = buf.getvalue()
    header = ' #     Column  Dtype  \n---    ------  -----  '
    assert header in res
    frame.info(verbose=True, buf=buf)
    buf.seek(0)
    lines = buf.readlines()
    assert len(lines) > 0
    for i, line in enumerate(lines):
        if start <= i < start + size:
            line_nr = f' {i - start} '
            assert line.startswith(line_nr)