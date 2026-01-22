import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.arrays import SparseArray
from pandas.tests.extension import base
@pytest.fixture(params=[0, np.nan])
def data_for_compare(request):
    return SparseArray([0, 0, np.nan, -2, -1, 4, 2, 3, 0, 0], fill_value=request.param)