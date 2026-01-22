from datetime import (
from functools import partial
from io import BytesIO
import os
import re
import numpy as np
import pytest
from pandas.compat import is_platform_windows
from pandas.compat._constants import PY310
from pandas.compat._optional import import_optional_dependency
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.excel import (
from pandas.io.excel._util import _writers
def test_comment_empty_line(self, path):
    df = DataFrame({'a': ['1', '#2'], 'b': ['2', '3']})
    df.to_excel(path, index=False)
    expected = DataFrame({'a': [1], 'b': [2]})
    result = pd.read_excel(path, comment='#')
    tm.assert_frame_equal(result, expected)