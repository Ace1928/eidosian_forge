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
def test_invalid_columns(self, path):
    write_frame = DataFrame({'A': [1, 1, 1], 'B': [2, 2, 2]})
    with pytest.raises(KeyError, match='Not all names specified'):
        write_frame.to_excel(path, sheet_name='test1', columns=['B', 'C'])
    with pytest.raises(KeyError, match="'passes columns are not ALL present dataframe'"):
        write_frame.to_excel(path, sheet_name='test1', columns=['C', 'D'])