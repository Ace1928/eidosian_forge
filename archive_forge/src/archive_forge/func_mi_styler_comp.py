import contextlib
import copy
import re
from textwrap import dedent
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.formats.style import (  # isort:skip
from pandas.io.formats.style_render import (
@pytest.fixture
def mi_styler_comp(mi_styler):
    mi_styler = mi_styler._copy(deepcopy=True)
    mi_styler.css = {**mi_styler.css, 'row': 'ROW', 'col': 'COL'}
    mi_styler.uuid_len = 5
    mi_styler.uuid = 'abcde'
    mi_styler.set_caption('capt')
    mi_styler.set_table_styles([{'selector': 'a', 'props': 'a:v;'}])
    mi_styler.hide(axis='columns')
    mi_styler.hide([('c0', 'c1_a')], axis='columns', names=True)
    mi_styler.hide(axis='index')
    mi_styler.hide([('i0', 'i1_a')], axis='index', names=True)
    mi_styler.set_table_attributes('class="box"')
    other = mi_styler.data.agg(['mean'])
    other.index = MultiIndex.from_product([[''], other.index])
    mi_styler.concat(other.style)
    mi_styler.format(na_rep='MISSING', precision=3)
    mi_styler.format_index(precision=2, axis=0)
    mi_styler.format_index(precision=4, axis=1)
    mi_styler.highlight_max(axis=None)
    mi_styler.map_index(lambda x: 'color: white;', axis=0)
    mi_styler.map_index(lambda x: 'color: black;', axis=1)
    mi_styler.set_td_classes(DataFrame([['a', 'b'], ['a', 'c']], index=mi_styler.index, columns=mi_styler.columns))
    mi_styler.set_tooltips(DataFrame([['a2', 'b2'], ['a2', 'c2']], index=mi_styler.index, columns=mi_styler.columns))
    return mi_styler