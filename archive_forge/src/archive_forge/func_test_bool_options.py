from textwrap import dedent
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
@pytest.mark.parametrize('option', ['hrules'])
def test_bool_options(styler, option):
    with option_context(f'styler.latex.{option}', False):
        latex_false = styler.to_latex()
    with option_context(f'styler.latex.{option}', True):
        latex_true = styler.to_latex()
    assert latex_false != latex_true