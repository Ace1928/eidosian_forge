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
def test_spam(self, spam_data, flavor_read_html):
    df1 = flavor_read_html(spam_data, match='.*Water.*')
    df2 = flavor_read_html(spam_data, match='Unit')
    assert_framelist_equal(df1, df2)
    assert df1[0].iloc[0, 0] == 'Proximates'
    assert df1[0].columns[0] == 'Nutrient'