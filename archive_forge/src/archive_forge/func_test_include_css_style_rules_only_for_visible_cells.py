from textwrap import (
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
def test_include_css_style_rules_only_for_visible_cells(styler_mi):
    result = styler_mi.set_uuid('').map(lambda v: 'color: blue;').hide(styler_mi.data.columns[1:], axis='columns').hide(styler_mi.data.index[1:], axis='index').to_html()
    expected_styles = dedent('        <style type="text/css">\n        #T__row0_col0 {\n          color: blue;\n        }\n        </style>\n        ')
    assert expected_styles in result