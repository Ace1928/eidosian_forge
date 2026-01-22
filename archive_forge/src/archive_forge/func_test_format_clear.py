import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import _str_escape
@pytest.mark.parametrize('func, attr, kwargs', [('format', '_display_funcs', {}), ('format_index', '_display_funcs_index', {'axis': 0}), ('format_index', '_display_funcs_columns', {'axis': 1})])
def test_format_clear(styler, func, attr, kwargs):
    assert (0, 0) not in getattr(styler, attr)
    getattr(styler, func)('{:.2f}', **kwargs)
    assert (0, 0) in getattr(styler, attr)
    getattr(styler, func)(**kwargs)
    assert (0, 0) not in getattr(styler, attr)