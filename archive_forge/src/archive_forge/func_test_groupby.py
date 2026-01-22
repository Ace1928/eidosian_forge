import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_groupby(self):
    df = SimpleDataFrameSubClass(DataFrame({'a': [1, 2, 3]}))
    for _, v in df.groupby('a'):
        assert type(v) is DataFrame