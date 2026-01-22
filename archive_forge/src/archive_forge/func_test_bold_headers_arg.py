from textwrap import (
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
def test_bold_headers_arg(styler):
    result = styler.to_html(bold_headers=True)
    assert 'th {\n  font-weight: bold;\n}' in result
    result = styler.to_html()
    assert 'th {\n  font-weight: bold;\n}' not in result