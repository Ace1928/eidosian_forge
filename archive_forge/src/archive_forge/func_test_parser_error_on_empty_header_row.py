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
def test_parser_error_on_empty_header_row(self, flavor_read_html):
    result = flavor_read_html(StringIO('\n                <table>\n                    <thead>\n                        <tr><th></th><th></tr>\n                        <tr><th>A</th><th>B</th></tr>\n                    </thead>\n                    <tbody>\n                        <tr><td>a</td><td>b</td></tr>\n                    </tbody>\n                </table>\n            '), header=[0, 1])
    expected = DataFrame([['a', 'b']], columns=MultiIndex.from_tuples([('Unnamed: 0_level_0', 'A'), ('Unnamed: 1_level_0', 'B')]))
    tm.assert_frame_equal(result[0], expected)