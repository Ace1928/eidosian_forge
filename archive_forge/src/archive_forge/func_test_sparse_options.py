from textwrap import dedent
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
@pytest.mark.parametrize('option, value', [('styler.sparse.index', True), ('styler.sparse.index', False), ('styler.sparse.columns', True), ('styler.sparse.columns', False)])
def test_sparse_options(df_ext, option, value):
    cidx = MultiIndex.from_tuples([('Z', 'a'), ('Z', 'b'), ('Y', 'c')])
    ridx = MultiIndex.from_tuples([('A', 'a'), ('A', 'b'), ('B', 'c')])
    df_ext.index, df_ext.columns = (ridx, cidx)
    styler = df_ext.style
    latex1 = styler.to_latex()
    with option_context(option, value):
        latex2 = styler.to_latex()
    assert (latex1 == latex2) is value