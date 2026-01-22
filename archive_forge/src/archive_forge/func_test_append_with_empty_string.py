import datetime
from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
def test_append_with_empty_string(setup_path):
    with ensure_clean_store(setup_path) as store:
        df = DataFrame({'x': ['a', 'b', 'c', 'd', 'e', 'f', '']})
        store.append('df', df[:-1], min_itemsize={'x': 1})
        store.append('df', df[-1:], min_itemsize={'x': 1})
        tm.assert_frame_equal(store.select('df'), df)