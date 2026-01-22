from __future__ import annotations
from datetime import (
from functools import partial
from io import BytesIO
import os
from pathlib import Path
import platform
import re
from urllib.error import URLError
from zipfile import BadZipFile
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_read_excel_bool_header_arg(self, read_ext):
    msg = 'Passing a bool to header is invalid'
    for arg in [True, False]:
        with pytest.raises(TypeError, match=msg):
            pd.read_excel('test1' + read_ext, header=arg)