from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_constructor_preserves_freq(self):
    dti = date_range('2016-01-01', periods=5)
    expected = dti.freq
    cat = Categorical(dti)
    result = cat.categories.freq
    assert expected == result