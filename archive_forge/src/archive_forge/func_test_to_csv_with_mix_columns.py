import csv
from io import StringIO
import os
import numpy as np
import pytest
from pandas.errors import ParserError
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
from pandas.io.common import get_handle
def test_to_csv_with_mix_columns(self):
    df = DataFrame({0: ['a', 'b', 'c'], 1: ['aa', 'bb', 'cc']})
    df['test'] = 'txt'
    assert df.to_csv() == df.to_csv(columns=[0, 1, 'test'])