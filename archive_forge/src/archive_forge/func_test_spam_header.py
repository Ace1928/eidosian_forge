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
def test_spam_header(self, spam_data, flavor_read_html):
    df = flavor_read_html(spam_data, match='.*Water.*', header=2)[0]
    assert df.columns[0] == 'Proximates'
    assert not df.empty