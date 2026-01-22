from textwrap import dedent
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
def test_parse_latex_css_conversion_option():
    css = [('command', 'option--latex--wrap')]
    expected = [('command', 'option--wrap')]
    result = _parse_latex_css_conversion(css)
    assert result == expected