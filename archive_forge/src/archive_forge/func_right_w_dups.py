from datetime import datetime
import numpy as np
import pytest
from pandas.errors import MergeError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
@pytest.fixture
def right_w_dups(right_no_dup):
    return concat([right_no_dup, DataFrame({'a': ['e'], 'c': ['moo']}, index=[3])]).set_index('a')