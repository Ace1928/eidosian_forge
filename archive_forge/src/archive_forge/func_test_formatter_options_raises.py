import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import _str_escape
def test_formatter_options_raises():
    msg = 'Value must be an instance of'
    with pytest.raises(ValueError, match=msg):
        with option_context('styler.format.formatter', ['bad', 'type']):
            DataFrame().style.to_latex()