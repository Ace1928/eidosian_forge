import string
import pytest
from pandas.errors import CSSWarning
import pandas._testing as tm
from pandas.io.formats.excel import (
@pytest.mark.parametrize('css,inherited,expected', [('font-weight: bold', '', {'font': {'bold': True}}), ('', 'font-weight: bold', {'font': {'bold': True}}), ('font-weight: bold', 'font-style: italic', {'font': {'bold': True, 'italic': True}}), ('font-style: normal', 'font-style: italic', {'font': {'italic': False}}), ('font-style: inherit', '', {}), ('font-style: normal; font-style: inherit', 'font-style: italic', {'font': {'italic': True}})])
def test_css_to_excel_inherited(css, inherited, expected):
    convert = CSSToExcelConverter(inherited)
    assert expected == convert(css)