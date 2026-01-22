from datetime import (
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
def test_at_categorical_integers(self):
    ci = CategoricalIndex([3, 4])
    arr = np.arange(4).reshape(2, 2)
    frame = DataFrame(arr, index=ci)
    for df in [frame, frame.T]:
        for key in [0, 1]:
            with pytest.raises(KeyError, match=str(key)):
                df.at[key, key]