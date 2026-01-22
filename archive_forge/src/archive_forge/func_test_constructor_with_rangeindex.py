from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_constructor_with_rangeindex(self):
    rng = Index(range(3))
    cat = Categorical(rng)
    tm.assert_index_equal(cat.categories, rng, exact=True)
    cat = Categorical([1, 2, 0], categories=rng)
    tm.assert_index_equal(cat.categories, rng, exact=True)