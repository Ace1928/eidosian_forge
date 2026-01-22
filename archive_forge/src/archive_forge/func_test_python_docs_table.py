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
@pytest.mark.network
@pytest.mark.single_cpu
def test_python_docs_table(self, python_docs, httpserver, flavor_read_html):
    httpserver.serve_content(content=python_docs)
    dfs = flavor_read_html(httpserver.url, match='Python')
    zz = [df.iloc[0, 0][0:4] for df in dfs]
    assert sorted(zz) == ['Pyth', 'What']