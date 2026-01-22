from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_categorical_1d_only(self):
    msg = '> 1 ndim Categorical are not supported at this time'
    with pytest.raises(NotImplementedError, match=msg):
        Categorical(np.array([list('abcd')]))