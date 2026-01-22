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
@pytest.mark.parametrize('displayed_only', [True, False])
def test_displayed_only_with_many_elements(self, displayed_only, flavor_read_html):
    html_table = '\n        <table>\n            <tr>\n                <th>A</th>\n                <th>B</th>\n            </tr>\n            <tr>\n                <td>1</td>\n                <td>2</td>\n            </tr>\n            <tr>\n                <td><span style="display:none"></span>4</td>\n                <td>5</td>\n            </tr>\n        </table>\n        '
    result = flavor_read_html(StringIO(html_table), displayed_only=displayed_only)[0]
    expected = DataFrame({'A': [1, 4], 'B': [2, 5]})
    tm.assert_frame_equal(result, expected)