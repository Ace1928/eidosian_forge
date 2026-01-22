import gc
import numpy as np
import pytest
from pandas import (
import matplotlib as mpl
from pandas.io.formats.style import Styler
@pytest.mark.parametrize('axis, gmap, expected', [(0, [1, 2], {(0, 0): [('background-color', '#fff7fb'), ('color', '#000000')], (1, 0): [('background-color', '#023858'), ('color', '#f1f1f1')], (0, 1): [('background-color', '#fff7fb'), ('color', '#000000')], (1, 1): [('background-color', '#023858'), ('color', '#f1f1f1')]}), (1, [1, 2], {(0, 0): [('background-color', '#fff7fb'), ('color', '#000000')], (1, 0): [('background-color', '#fff7fb'), ('color', '#000000')], (0, 1): [('background-color', '#023858'), ('color', '#f1f1f1')], (1, 1): [('background-color', '#023858'), ('color', '#f1f1f1')]}), (None, np.array([[2, 1], [1, 2]]), {(0, 0): [('background-color', '#023858'), ('color', '#f1f1f1')], (1, 0): [('background-color', '#fff7fb'), ('color', '#000000')], (0, 1): [('background-color', '#fff7fb'), ('color', '#000000')], (1, 1): [('background-color', '#023858'), ('color', '#f1f1f1')]})])
def test_background_gradient_gmap_array(styler_blank, axis, gmap, expected):
    result = styler_blank.background_gradient(axis=axis, gmap=gmap)._compute().ctx
    assert result == expected