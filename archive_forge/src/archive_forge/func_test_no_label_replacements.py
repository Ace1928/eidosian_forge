import re
import sys
import numpy as np
import pytest
from matplotlib import _preprocess_data
from matplotlib.axes import Axes
from matplotlib.testing import subprocess_run_for_testing
from matplotlib.testing.decorators import check_figures_equal
def test_no_label_replacements():
    """Test with "label_namer=None" -> no label replacement at all."""

    @_preprocess_data(replace_names=['x', 'y'], label_namer=None)
    def func_no_label(ax, x, y, ls='x', label=None, w='xyz'):
        return f'x: {list(x)}, y: {list(y)}, ls: {ls}, w: {w}, label: {label}'
    data = {'a': [1, 2], 'b': [8, 9], 'w': 'NOT'}
    assert func_no_label(None, 'a', 'b', data=data) == 'x: [1, 2], y: [8, 9], ls: x, w: xyz, label: None'
    assert func_no_label(None, x='a', y='b', data=data) == 'x: [1, 2], y: [8, 9], ls: x, w: xyz, label: None'
    assert func_no_label(None, 'a', 'b', label='', data=data) == 'x: [1, 2], y: [8, 9], ls: x, w: xyz, label: '
    assert func_no_label(None, 'a', 'b', label='text', data=data) == 'x: [1, 2], y: [8, 9], ls: x, w: xyz, label: text'