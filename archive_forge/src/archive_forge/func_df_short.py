import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.fixture
def df_short():
    """Short dataframe for testing table/tabular/longtable LaTeX env."""
    return DataFrame({'a': [1, 2], 'b': ['b1', 'b2']})