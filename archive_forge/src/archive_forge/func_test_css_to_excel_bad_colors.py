import string
import pytest
from pandas.errors import CSSWarning
import pandas._testing as tm
from pandas.io.formats.excel import (
@pytest.mark.parametrize('input_color', [None, 'not-a-color'])
def test_css_to_excel_bad_colors(input_color):
    css = f'border-top-color: {input_color}; border-right-color: {input_color}; border-bottom-color: {input_color}; border-left-color: {input_color}; background-color: {input_color}; color: {input_color}'
    expected = {}
    if input_color is not None:
        expected['fill'] = {'patternType': 'solid'}
    with tm.assert_produces_warning(CSSWarning):
        convert = CSSToExcelConverter()
        assert expected == convert(css)