import numpy as np
import pytest
from pandas import DataFrame
from pandas.io.formats.style import Styler
@pytest.mark.parametrize('ttips', [DataFrame(data=[['Min', 'Max'], [np.nan, '']], columns=['A', 'C'], index=['x', 'y']), DataFrame(data=[['Max', 'Min', 'Bad-Col']], columns=['C', 'A', 'D'], index=['x'])])
def test_tooltip_render(ttips, styler):
    result = styler.set_tooltips(ttips).to_html()
    assert '#T_ .pd-t {\n  visibility: hidden;\n' in result
    assert '#T_ #T__row0_col0:hover .pd-t {\n  visibility: visible;\n}' in result
    assert '#T_ #T__row0_col0 .pd-t::after {\n  content: "Min";\n}' in result
    assert 'class="data row0 col0" >0<span class="pd-t"></span></td>' in result
    assert '#T_ #T__row0_col2:hover .pd-t {\n  visibility: visible;\n}' in result
    assert '#T_ #T__row0_col2 .pd-t::after {\n  content: "Max";\n}' in result
    assert 'class="data row0 col2" >2<span class="pd-t"></span></td>' in result
    assert '#T_ #T__row1_col0:hover .pd-t {\n  visibility: visible;\n}' not in result
    assert '#T_ #T__row1_col1:hover .pd-t {\n  visibility: visible;\n}' not in result
    assert '#T_ #T__row0_col1:hover .pd-t {\n  visibility: visible;\n}' not in result
    assert '#T_ #T__row1_col2:hover .pd-t {\n  visibility: visible;\n}' not in result
    assert 'Bad-Col' not in result