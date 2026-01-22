from datetime import datetime
import numpy as np
import pytest
from pandas.errors import MergeError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
@pytest.fixture
def right_no_dup():
    return DataFrame({'a': ['a', 'b', 'c', 'd', 'e'], 'c': ['meow', 'bark', 'um... weasel noise?', 'nay', 'chirp']}, index=range(5)).set_index('a')