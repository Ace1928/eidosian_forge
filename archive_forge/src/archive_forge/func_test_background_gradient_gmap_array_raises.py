import gc
import numpy as np
import pytest
from pandas import (
import matplotlib as mpl
from pandas.io.formats.style import Styler
@pytest.mark.parametrize('gmap, axis', [([1, 2, 3], 0), ([1, 2], 1), (np.array([[1, 2], [1, 2]]), None)])
def test_background_gradient_gmap_array_raises(gmap, axis):
    df = DataFrame([[0, 0, 0], [0, 0, 0]])
    msg = "supplied 'gmap' is not correct shape"
    with pytest.raises(ValueError, match=msg):
        df.style.background_gradient(gmap=gmap, axis=axis)._compute()