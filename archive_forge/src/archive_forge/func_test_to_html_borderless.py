from collections.abc import Iterator
from functools import partial
from io import (
import os
from pathlib import Path
import re
import threading
from urllib.error import URLError
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.io.common import file_path_to_url
def test_to_html_borderless(self):
    df = DataFrame([{'A': 1, 'B': 2}])
    out_border_default = df.to_html()
    out_border_true = df.to_html(border=True)
    out_border_explicit_default = df.to_html(border=1)
    out_border_nondefault = df.to_html(border=2)
    out_border_zero = df.to_html(border=0)
    out_border_false = df.to_html(border=False)
    assert ' border="1"' in out_border_default
    assert out_border_true == out_border_default
    assert out_border_default == out_border_explicit_default
    assert out_border_default != out_border_nondefault
    assert ' border="2"' in out_border_nondefault
    assert ' border="0"' not in out_border_zero
    assert ' border' not in out_border_false
    assert out_border_zero == out_border_false