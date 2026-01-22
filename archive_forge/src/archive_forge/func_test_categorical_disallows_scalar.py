from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_categorical_disallows_scalar(self):
    with pytest.raises(TypeError, match='Categorical input must be list-like'):
        Categorical('A', categories=['A', 'B'])