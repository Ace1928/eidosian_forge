from io import StringIO
import re
from string import ascii_uppercase as uppercase
import sys
import textwrap
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
@pytest.fixture
def duplicate_columns_frame():
    """Dataframe with duplicate column names."""
    return DataFrame(np.random.randn(1500, 4), columns=['a', 'a', 'b', 'b'])