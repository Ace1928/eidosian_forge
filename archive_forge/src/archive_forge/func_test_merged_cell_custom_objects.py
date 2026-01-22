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
def test_merged_cell_custom_objects(self, path):
    mi = MultiIndex.from_tuples([(pd.Period('2018'), pd.Period('2018Q1')), (pd.Period('2018'), pd.Period('2018Q2'))])
    expected = DataFrame(np.ones((2, 2), dtype='int64'), columns=mi)
    expected.to_excel(path)
    result = pd.read_excel(path, header=[0, 1], index_col=0)
    expected.columns = expected.columns.set_levels([[str(i) for i in mi.levels[0]], [str(i) for i in mi.levels[1]]], level=[0, 1])
    tm.assert_frame_equal(result, expected)