import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_error_no_prefix_multi_assignment():
    dummies = DataFrame({'a': [1, 0, 1], 'b': [0, 1, 1]})
    with pytest.raises(ValueError, match='Dummy DataFrame contains multi-assignment\\(s\\); First instance in row: 2'):
        from_dummies(dummies)