import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
@pytest.mark.parametrize('case', [0.5, 'xxx'])
@pytest.mark.parametrize('method', ['intersection', 'union', 'difference', 'symmetric_difference'])
def test_set_ops_error_cases(idx, case, sort, method):
    msg = 'Input must be Index or array-like'
    with pytest.raises(TypeError, match=msg):
        getattr(idx, method)(case, sort=sort)