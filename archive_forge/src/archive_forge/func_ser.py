from itertools import chain
import operator
import numpy as np
import pytest
from pandas._libs.algos import (
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
@pytest.fixture
def ser():
    return Series([1, 3, 4, 2, np.nan, 2, 1, 5, np.nan, 3])