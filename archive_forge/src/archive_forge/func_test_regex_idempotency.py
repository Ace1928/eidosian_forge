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
@pytest.mark.slow
def test_regex_idempotency(self, banklist_data, flavor_read_html):
    url = banklist_data
    dfs = flavor_read_html(file_path_to_url(os.path.abspath(url)), match=re.compile(re.compile('Florida')), attrs={'id': 'table'})
    assert isinstance(dfs, list)
    for df in dfs:
        assert isinstance(df, DataFrame)