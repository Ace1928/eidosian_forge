import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('n', [-1, 1.5])
def test_groupby_sample_invalid_n_raises(n):
    df = DataFrame({'a': [1, 2], 'b': [1, 2]})
    if n < 0:
        msg = 'A negative number of rows requested. Please provide `n` >= 0.'
    else:
        msg = 'Only integers accepted as `n` values'
    with pytest.raises(ValueError, match=msg):
        df.groupby('a').sample(n=n)
    with pytest.raises(ValueError, match=msg):
        df.groupby('a')['b'].sample(n=n)