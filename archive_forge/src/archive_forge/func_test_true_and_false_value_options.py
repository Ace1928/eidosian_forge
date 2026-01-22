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
def test_true_and_false_value_options(self, path):
    df = DataFrame([['foo', 'bar']], columns=['col1', 'col2'], dtype=object)
    with option_context('future.no_silent_downcasting', True):
        expected = df.replace({'foo': True, 'bar': False}).astype('bool')
    df.to_excel(path)
    read_frame = pd.read_excel(path, true_values=['foo'], false_values=['bar'], index_col=0)
    tm.assert_frame_equal(read_frame, expected)