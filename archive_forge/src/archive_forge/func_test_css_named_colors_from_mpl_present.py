import string
import pytest
from pandas.errors import CSSWarning
import pandas._testing as tm
from pandas.io.formats.excel import (
def test_css_named_colors_from_mpl_present():
    mpl_colors = pytest.importorskip('matplotlib.colors')
    pd_colors = CSSToExcelConverter.NAMED_COLORS
    for name, color in mpl_colors.CSS4_COLORS.items():
        assert name in pd_colors and pd_colors[name] == color[1:]