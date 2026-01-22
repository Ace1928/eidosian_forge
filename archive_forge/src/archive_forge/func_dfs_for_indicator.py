from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
@pytest.fixture
def dfs_for_indicator():
    df1 = DataFrame({'col1': [0, 1], 'col_conflict': [1, 2], 'col_left': ['a', 'b']})
    df2 = DataFrame({'col1': [1, 2, 3, 4, 5], 'col_conflict': [1, 2, 3, 4, 5], 'col_right': [2, 2, 2, 2, 2]})
    return (df1, df2)