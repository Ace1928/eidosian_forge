from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('ordered', [None, True, False])
def test_construction_with_ordered(self, ordered):
    cat = Categorical([0, 1, 2], ordered=ordered)
    assert cat.ordered == bool(ordered)