import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_quantile_out_of_bounds_q_raises():
    df = DataFrame({'a': [0, 0, 0, 1, 1, 1], 'b': range(6)})
    g = df.groupby([0, 0, 0, 1, 1, 1])
    with pytest.raises(ValueError, match="Got '50.0' instead"):
        g.quantile(50)
    with pytest.raises(ValueError, match="Got '-1.0' instead"):
        g.quantile(-1)