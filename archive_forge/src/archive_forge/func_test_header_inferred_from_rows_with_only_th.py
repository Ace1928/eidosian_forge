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
def test_header_inferred_from_rows_with_only_th(self, flavor_read_html):
    result = flavor_read_html(StringIO('\n            <table>\n                <tr>\n                    <th>A</th>\n                    <th>B</th>\n                </tr>\n                <tr>\n                    <th>a</th>\n                    <th>b</th>\n                </tr>\n                <tr>\n                    <td>1</td>\n                    <td>2</td>\n                </tr>\n            </table>\n        '))[0]
    columns = MultiIndex(levels=[['A', 'B'], ['a', 'b']], codes=[[0, 1], [0, 1]])
    expected = DataFrame(data=[[1, 2]], columns=columns)
    tm.assert_frame_equal(result, expected)