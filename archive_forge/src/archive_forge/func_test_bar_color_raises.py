import gc
import numpy as np
import pytest
from pandas import (
import matplotlib as mpl
from pandas.io.formats.style import Styler
def test_bar_color_raises(df):
    msg = '`color` must be string or list or tuple of 2 strings'
    with pytest.raises(ValueError, match=msg):
        df.style.bar(color={'a', 'b'}).to_html()
    with pytest.raises(ValueError, match=msg):
        df.style.bar(color=['a', 'b', 'c']).to_html()
    msg = '`color` and `cmap` cannot both be given'
    with pytest.raises(ValueError, match=msg):
        df.style.bar(color='something', cmap='something else').to_html()