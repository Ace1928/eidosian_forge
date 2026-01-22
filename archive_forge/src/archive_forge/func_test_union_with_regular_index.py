import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
def test_union_with_regular_index(idx, using_infer_string):
    other = Index(['A', 'B', 'C'])
    result = other.union(idx)
    assert ('foo', 'one') in result
    assert 'B' in result
    if using_infer_string:
        with pytest.raises(NotImplementedError, match='Can only union'):
            idx.union(other)
    else:
        msg = 'The values in the array are unorderable'
        with tm.assert_produces_warning(RuntimeWarning, match=msg):
            result2 = idx.union(other)
        assert not result.equals(result2)