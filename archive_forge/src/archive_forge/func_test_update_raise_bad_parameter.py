import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('bad_kwarg, exception, msg', [({'errors': 'something'}, ValueError, 'The parameter errors must.*'), ({'join': 'inner'}, NotImplementedError, 'Only left join is supported')])
def test_update_raise_bad_parameter(self, bad_kwarg, exception, msg):
    df = DataFrame([[1.5, 1, 3.0]])
    with pytest.raises(exception, match=msg):
        df.update(df, **bad_kwarg)