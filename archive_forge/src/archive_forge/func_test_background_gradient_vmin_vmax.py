import gc
import numpy as np
import pytest
from pandas import (
import matplotlib as mpl
from pandas.io.formats.style import Styler
def test_background_gradient_vmin_vmax():
    df = DataFrame(range(5))
    ctx = df.style.background_gradient(vmin=1, vmax=3)._compute().ctx
    assert ctx[0, 0] == ctx[1, 0]
    assert ctx[4, 0] == ctx[3, 0]