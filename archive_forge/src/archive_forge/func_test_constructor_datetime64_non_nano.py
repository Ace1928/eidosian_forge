from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_constructor_datetime64_non_nano(self):
    categories = np.arange(10).view('M8[D]')
    values = categories[::2].copy()
    cat = Categorical(values, categories=categories)
    assert (cat == values).all()