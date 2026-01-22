import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
def test_highlight_null(styler):
    result = styler.highlight_null()._compute().ctx
    expected = {(1, 0): [('background-color', 'red')], (1, 1): [('background-color', 'red')]}
    assert result == expected