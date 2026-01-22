import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
@pytest.fixture(params=[['outer'], ['outer', 'inner']])
def df_idx(request, df_none):
    levels = request.param
    return df_none.set_index(levels)