import numpy as np
import pytest
from pandas import DataFrame
import pandas._testing as tm
@pytest.fixture
def df1():
    return DataFrame({'outer': [1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4], 'inner': [1, 2, 3, 1, 2, 3, 4, 1, 2, 1, 2], 'v1': np.linspace(0, 1, 11)})