import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_compare_frame_raises(self, comparison_op):
    op = comparison_op
    cat = Categorical(['a', 'b', 2, 'a'])
    df = DataFrame(cat)
    msg = 'Unable to coerce to Series, length must be 1: given 4'
    with pytest.raises(ValueError, match=msg):
        op(cat, df)