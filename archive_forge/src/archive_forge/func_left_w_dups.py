from datetime import datetime
import numpy as np
import pytest
from pandas.errors import MergeError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
@pytest.fixture
def left_w_dups(left_no_dup):
    return concat([left_no_dup, DataFrame({'a': ['a'], 'b': ['cow']}, index=[3])], sort=True)