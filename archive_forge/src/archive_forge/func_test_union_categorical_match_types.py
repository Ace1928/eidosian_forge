import numpy as np
import pytest
from pandas.core.dtypes.concat import union_categoricals
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_union_categorical_match_types(self):
    s = Categorical([0, 1.2, 2])
    s2 = Categorical([2, 3, 4])
    msg = 'dtype of categories must be the same'
    with pytest.raises(TypeError, match=msg):
        union_categoricals([s, s2])