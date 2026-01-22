from datetime import (
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import IncompatibleFrequency
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_unequal_categorical_comparison_raises_type_error(self):
    cat = Series(Categorical(list('abc')))
    msg = 'can only compare equality or not'
    with pytest.raises(TypeError, match=msg):
        cat > 'b'
    cat = Series(Categorical(list('abc'), ordered=False))
    with pytest.raises(TypeError, match=msg):
        cat > 'b'
    cat = Series(Categorical(list('abc'), ordered=True))
    msg = 'Invalid comparison between dtype=category and str'
    with pytest.raises(TypeError, match=msg):
        cat < 'd'
    with pytest.raises(TypeError, match=msg):
        cat > 'd'
    with pytest.raises(TypeError, match=msg):
        'd' < cat
    with pytest.raises(TypeError, match=msg):
        'd' > cat
    tm.assert_series_equal(cat == 'd', Series([False, False, False]))
    tm.assert_series_equal(cat != 'd', Series([True, True, True]))