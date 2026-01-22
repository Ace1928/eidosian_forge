from textwrap import (
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
@pytest.mark.parametrize('index', [True, False])
@pytest.mark.parametrize('columns', [True, False])
def test_map_header_cell_ids(styler, index, columns):
    func = lambda v: 'attr: val;'
    styler.uuid, styler.cell_ids = ('', False)
    if index:
        styler.map_index(func, axis='index')
    if columns:
        styler.map_index(func, axis='columns')
    result = styler.to_html()
    assert '<td class="data row0 col0" >2.610000</td>' in result
    assert '<td class="data row1 col0" >2.690000</td>' in result
    assert ('<th id="T__level0_row0" class="row_heading level0 row0" >a</th>' in result) is index
    assert ('<th id="T__level0_row1" class="row_heading level0 row1" >b</th>' in result) is index
    assert ('#T__level0_row0, #T__level0_row1 {\n  attr: val;\n}' in result) is index
    assert ('<th id="T__level0_col0" class="col_heading level0 col0" >A</th>' in result) is columns
    assert ('#T__level0_col0 {\n  attr: val;\n}' in result) is columns