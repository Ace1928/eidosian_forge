from datetime import (
import itertools
import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_raise_invalid_sortorder():
    levels = [[0, 1], [0, 1, 2]]
    MultiIndex(levels=levels, codes=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]], sortorder=2)
    with pytest.raises(ValueError, match='.* sortorder 2 with lexsort_depth 1.*'):
        MultiIndex(levels=levels, codes=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 2, 1]], sortorder=2)
    with pytest.raises(ValueError, match='.* sortorder 1 with lexsort_depth 0.*'):
        MultiIndex(levels=levels, codes=[[0, 0, 1, 0, 1, 1], [0, 1, 0, 2, 2, 1]], sortorder=1)