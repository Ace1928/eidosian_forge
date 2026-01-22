from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_validate_ordered(self):
    exp_msg = "'ordered' must either be 'True' or 'False'"
    exp_err = TypeError
    ordered = np.array([0, 1, 2])
    with pytest.raises(exp_err, match=exp_msg):
        Categorical([1, 2, 3], ordered=ordered)
    with pytest.raises(exp_err, match=exp_msg):
        Categorical.from_codes([0, 0, 1], categories=['a', 'b', 'c'], ordered=ordered)