import re
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.util import _test_decorators as td
def test_format_type(tmp_path, setup_path):
    df = DataFrame({'A': [1, 2]})
    with HDFStore(tmp_path / setup_path) as store:
        store.put('a', df, format='fixed')
        store.put('b', df, format='table')
        assert store.get_storer('a').format_type == 'fixed'
        assert store.get_storer('b').format_type == 'table'