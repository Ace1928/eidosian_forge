import re
import sys
import numpy as np
import pytest
from matplotlib import _preprocess_data
from matplotlib.axes import Axes
from matplotlib.testing import subprocess_run_for_testing
from matplotlib.testing.decorators import check_figures_equal
@pytest.mark.parametrize('func', all_funcs, ids=all_func_ids)
def test_function_call_without_data(func):
    """Test without data -> no replacements."""
    assert func(None, 'x', 'y') == "x: ['x'], y: ['y'], ls: x, w: xyz, label: None"
    assert func(None, x='x', y='y') == "x: ['x'], y: ['y'], ls: x, w: xyz, label: None"
    assert func(None, 'x', 'y', label='') == "x: ['x'], y: ['y'], ls: x, w: xyz, label: "
    assert func(None, 'x', 'y', label='text') == "x: ['x'], y: ['y'], ls: x, w: xyz, label: text"
    assert func(None, x='x', y='y', label='') == "x: ['x'], y: ['y'], ls: x, w: xyz, label: "
    assert func(None, x='x', y='y', label='text') == "x: ['x'], y: ['y'], ls: x, w: xyz, label: text"