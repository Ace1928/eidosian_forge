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
def test_thead_without_tr(self, flavor_read_html):
    """
        Ensure parser adds <tr> within <thead> on malformed HTML.
        """
    result = flavor_read_html(StringIO('<table>\n            <thead>\n                <tr>\n                    <th>Country</th>\n                    <th>Municipality</th>\n                    <th>Year</th>\n                </tr>\n            </thead>\n            <tbody>\n                <tr>\n                    <td>Ukraine</td>\n                    <th>Odessa</th>\n                    <td>1944</td>\n                </tr>\n            </tbody>\n        </table>'))[0]
    expected = DataFrame(data=[['Ukraine', 'Odessa', 1944]], columns=['Country', 'Municipality', 'Year'])
    tm.assert_frame_equal(result, expected)