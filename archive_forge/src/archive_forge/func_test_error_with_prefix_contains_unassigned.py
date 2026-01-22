import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_error_with_prefix_contains_unassigned(dummies_with_unassigned):
    with pytest.raises(ValueError, match='Dummy DataFrame contains unassigned value\\(s\\); First instance in row: 2'):
        from_dummies(dummies_with_unassigned, sep='_')