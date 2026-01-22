from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_constructor_with_existing_categories(self):
    c0 = Categorical(['a', 'b', 'c', 'a'])
    c1 = Categorical(['a', 'b', 'c', 'a'], categories=['b', 'c'])
    c2 = Categorical(c0, categories=c1.categories)
    tm.assert_categorical_equal(c1, c2)
    c3 = Categorical(Series(c0), categories=c1.categories)
    tm.assert_categorical_equal(c1, c3)