import string
import pytest
from pandas.errors import CSSWarning
import pandas._testing as tm
from pandas.io.formats.excel import (
def test_css_to_excel_multiple():
    convert = CSSToExcelConverter()
    actual = convert('\n        font-weight: bold;\n        text-decoration: underline;\n        color: red;\n        border-width: thin;\n        text-align: center;\n        vertical-align: top;\n        unused: something;\n    ')
    assert {'font': {'bold': True, 'underline': 'single', 'color': 'FF0000'}, 'border': {'top': {'style': 'thin'}, 'right': {'style': 'thin'}, 'bottom': {'style': 'thin'}, 'left': {'style': 'thin'}}, 'alignment': {'horizontal': 'center', 'vertical': 'top'}} == actual