import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas import (
import pandas._testing as tm
def test_iter_python_types(self):
    cat = Categorical([1, 2])
    assert isinstance(next(iter(cat)), int)
    assert isinstance(cat.tolist()[0], int)