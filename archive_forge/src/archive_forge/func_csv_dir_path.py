from __future__ import annotations
import os
import pytest
from pandas.compat._optional import VERSIONS
from pandas import (
import pandas._testing as tm
@pytest.fixture
def csv_dir_path(datapath):
    """
    The directory path to the data files needed for parser tests.
    """
    return datapath('io', 'parser', 'data')