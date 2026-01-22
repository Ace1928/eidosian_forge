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
def test_to_excel_output_encoding(self, ext):
    df = DataFrame([['ƒ', 'Ɠ', 'Ɣ'], ['ƕ', 'Ɩ', 'Ɨ']], index=['Aƒ', 'B'], columns=['XƓ', 'Y', 'Z'])
    with tm.ensure_clean('__tmp_to_excel_float_format__.' + ext) as filename:
        df.to_excel(filename, sheet_name='TestSheet')
        result = pd.read_excel(filename, sheet_name='TestSheet', index_col=0)
        tm.assert_frame_equal(result, df)