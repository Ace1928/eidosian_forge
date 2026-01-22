from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_from_codes_with_nan_code(self):
    codes = [1, 2, np.nan]
    dtype = CategoricalDtype(categories=['a', 'b', 'c'])
    with pytest.raises(ValueError, match='codes need to be array-like integers'):
        Categorical.from_codes(codes, categories=dtype.categories)
    with pytest.raises(ValueError, match='codes need to be array-like integers'):
        Categorical.from_codes(codes, dtype=dtype)