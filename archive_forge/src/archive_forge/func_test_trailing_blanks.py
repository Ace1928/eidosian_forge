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
def test_trailing_blanks(self, read_ext):
    """
        Sheets can contain blank cells with no data. Some of our readers
        were including those cells, creating many empty rows and columns
        """
    file_name = 'trailing_blanks' + read_ext
    result = pd.read_excel(file_name)
    assert result.shape == (3, 3)