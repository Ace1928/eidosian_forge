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
def test_to_string_with_datetime64_hourformatter(self):
    x = DataFrame({'hod': to_datetime(['10:10:10.100', '12:12:12.120'], format='%H:%M:%S.%f')})

    def format_func(x):
        return x.strftime('%H:%M')
    result = x.to_string(formatters={'hod': format_func})
    expected = dedent('            hod\n            0 10:10\n            1 12:12')
    assert result.strip() == expected