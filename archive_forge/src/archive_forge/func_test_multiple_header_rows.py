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
def test_multiple_header_rows(self, flavor_read_html):
    expected_df = DataFrame(data=[('Hillary', 68, 'D'), ('Bernie', 74, 'D'), ('Donald', 69, 'R')])
    expected_df.columns = [['Unnamed: 0_level_0', 'Age', 'Party'], ['Name', 'Unnamed: 1_level_1', 'Unnamed: 2_level_1']]
    html = expected_df.to_html(index=False)
    html_df = flavor_read_html(StringIO(html))[0]
    tm.assert_frame_equal(expected_df, html_df)