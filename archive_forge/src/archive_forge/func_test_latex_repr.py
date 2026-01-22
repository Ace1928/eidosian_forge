from datetime import (
from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.io.formats.format as fmt
def test_latex_repr(self):
    pytest.importorskip('jinja2')
    expected = '\\begin{tabular}{llll}\n\\toprule\n & 0 & 1 & 2 \\\\\n\\midrule\n0 & $\\alpha$ & b & c \\\\\n1 & 1 & 2 & 3 \\\\\n\\bottomrule\n\\end{tabular}\n'
    with option_context('styler.format.escape', None, 'styler.render.repr', 'latex'):
        df = DataFrame([['$\\alpha$', 'b', 'c'], [1, 2, 3]])
        result = df._repr_latex_()
        assert result == expected
    assert df._repr_latex_() is None