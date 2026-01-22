import gc
import numpy as np
import pytest
from pandas import (
import matplotlib as mpl
from pandas.io.formats.style import Styler
def test_background_gradient_gmap_wrong_series(styler_blank):
    msg = "'gmap' is a Series but underlying data for operations is a DataFrame"
    gmap = Series([1, 2], index=['X', 'Y'])
    with pytest.raises(ValueError, match=msg):
        styler_blank.background_gradient(gmap=gmap, axis=None)._compute()