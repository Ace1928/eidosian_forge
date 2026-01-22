from io import BytesIO
import logging
import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import DataFrame
import pandas._testing as tm
from pandas.io.feather_format import read_feather
from pandas.io.parsers import read_csv
@pytest.fixture
def tips_df(datapath):
    """DataFrame with the tips dataset."""
    return read_csv(datapath('io', 'data', 'csv', 'tips.csv'))