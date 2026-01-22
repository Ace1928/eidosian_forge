from datetime import datetime
from itertools import product
import numpy as np
import pytest
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.fixture()
def multiindex_df():
    levels = [['A', ''], ['B', 'b']]
    return DataFrame([[0, 2], [1, 3]], columns=MultiIndex.from_tuples(levels))