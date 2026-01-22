from datetime import (
import numpy as np
import pytest
from pandas._libs.algos import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('method,exp', [('dense', [[1.0, 1.0, 1.0], [1.0, 0.5, 2.0 / 3], [1.0, 0.5, 1.0 / 3]]), ('min', [[1.0 / 3, 1.0, 1.0], [1.0 / 3, 1.0 / 3, 2.0 / 3], [1.0 / 3, 1.0 / 3, 1.0 / 3]]), ('max', [[1.0, 1.0, 1.0], [1.0, 2.0 / 3, 2.0 / 3], [1.0, 2.0 / 3, 1.0 / 3]]), ('average', [[2.0 / 3, 1.0, 1.0], [2.0 / 3, 0.5, 2.0 / 3], [2.0 / 3, 0.5, 1.0 / 3]]), ('first', [[1.0 / 3, 1.0, 1.0], [2.0 / 3, 1.0 / 3, 2.0 / 3], [3.0 / 3, 2.0 / 3, 1.0 / 3]])])
def test_rank_pct_true(self, method, exp):
    df = DataFrame([[2012, 66, 3], [2012, 65, 2], [2012, 65, 1]])
    result = df.rank(method=method, pct=True)
    expected = DataFrame(exp)
    tm.assert_frame_equal(result, expected)