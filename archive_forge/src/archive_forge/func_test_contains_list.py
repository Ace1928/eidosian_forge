import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_contains_list(self):
    idx = CategoricalIndex([1, 2, 3])
    assert 'a' not in idx
    with pytest.raises(TypeError, match='unhashable type'):
        ['a'] in idx
    with pytest.raises(TypeError, match='unhashable type'):
        ['a', 'b'] in idx