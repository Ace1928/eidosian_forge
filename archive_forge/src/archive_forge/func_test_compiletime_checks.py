import re
import sys
import numpy as np
import pytest
from matplotlib import _preprocess_data
from matplotlib.axes import Axes
from matplotlib.testing import subprocess_run_for_testing
from matplotlib.testing.decorators import check_figures_equal
def test_compiletime_checks():
    """Test decorator invocations -> no replacements."""

    def func(ax, x, y):
        pass

    def func_args(ax, x, y, *args):
        pass

    def func_kwargs(ax, x, y, **kwargs):
        pass

    def func_no_ax_args(*args, **kwargs):
        pass
    _preprocess_data(replace_names=['x', 'y'])(func)
    _preprocess_data(replace_names=['x', 'y'])(func_kwargs)
    _preprocess_data(replace_names=['x', 'y'])(func_args)
    with pytest.raises(AssertionError):
        _preprocess_data(replace_names=['x', 'y', 'z'])(func_args)
    _preprocess_data(replace_names=[], label_namer=None)(func)
    _preprocess_data(replace_names=[], label_namer=None)(func_args)
    _preprocess_data(replace_names=[], label_namer=None)(func_kwargs)
    _preprocess_data(replace_names=[], label_namer=None)(func_no_ax_args)
    with pytest.raises(AssertionError):
        _preprocess_data(label_namer='z')(func)
    with pytest.raises(AssertionError):
        _preprocess_data(label_namer='z')(func_args)