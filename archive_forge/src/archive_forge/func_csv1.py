from __future__ import annotations
import os
import pytest
from pandas.compat._optional import VERSIONS
from pandas import (
import pandas._testing as tm
@pytest.fixture
def csv1(datapath):
    """
    The path to the data file "test1.csv" needed for parser tests.
    """
    return os.path.join(datapath('io', 'data', 'csv'), 'test1.csv')