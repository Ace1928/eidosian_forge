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
def test_to_string_with_formatters_unicode(self):
    df = DataFrame({'c/σ': [1, 2, 3]})
    result = df.to_string(formatters={'c/σ': str})
    expected = dedent('              c/σ\n            0   1\n            1   2\n            2   3')
    assert result == expected

    def test_to_string_index_formatter(self):
        df = DataFrame([range(5), range(5, 10), range(10, 15)])
        rs = df.to_string(formatters={'__index__': lambda x: 'abc'[x]})
        xp = dedent('                0   1   2   3   4\n            a   0   1   2   3   4\n            b   5   6   7   8   9\n            c  10  11  12  13  14            ')
        assert rs == xp