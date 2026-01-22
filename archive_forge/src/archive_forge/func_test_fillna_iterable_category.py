import collections
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('named', [True, False])
def test_fillna_iterable_category(self, named):
    if named:
        Point = collections.namedtuple('Point', 'x y')
    else:
        Point = lambda *args: args
    cat = Categorical(np.array([Point(0, 0), Point(0, 1), None], dtype=object))
    result = cat.fillna(Point(0, 0))
    expected = Categorical([Point(0, 0), Point(0, 1), Point(0, 0)])
    tm.assert_categorical_equal(result, expected)
    cat = Categorical(np.array([Point(1, 0), Point(0, 1), None], dtype=object))
    msg = 'Cannot setitem on a Categorical with a new category'
    with pytest.raises(TypeError, match=msg):
        cat.fillna(Point(0, 0))