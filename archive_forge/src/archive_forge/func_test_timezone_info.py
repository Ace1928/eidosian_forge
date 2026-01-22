from datetime import (
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.groupby.ops import BinGrouper
def test_timezone_info(self):
    df = DataFrame({'a': [1], 'b': [datetime.now(pytz.utc)]})
    assert df['b'][0].tzinfo == pytz.utc
    df = DataFrame({'a': [1, 2, 3]})
    df['b'] = datetime.now(pytz.utc)
    assert df['b'][0].tzinfo == pytz.utc