import gc
import numpy as np
import pytest
from pandas import (
import matplotlib as mpl
from pandas.io.formats.style import Styler
@pytest.mark.parametrize('cmap, expected', [('PuBu', {(4, 5): [('background-color', '#86b0d3'), ('color', '#000000')], (4, 6): [('background-color', '#83afd3'), ('color', '#f1f1f1')]}), ('YlOrRd', {(4, 8): [('background-color', '#fd913e'), ('color', '#000000')], (4, 9): [('background-color', '#fd8f3d'), ('color', '#f1f1f1')]}), (None, {(7, 0): [('background-color', '#48c16e'), ('color', '#f1f1f1')], (7, 1): [('background-color', '#4cc26c'), ('color', '#000000')]})])
def test_text_color_threshold(cmap, expected):
    df = DataFrame(np.arange(100).reshape(10, 10))
    result = df.style.background_gradient(cmap=cmap, axis=None)._compute().ctx
    for k in expected.keys():
        assert result[k] == expected[k]