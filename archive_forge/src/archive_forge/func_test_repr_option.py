from textwrap import dedent
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
def test_repr_option(styler):
    assert '<style' in styler._repr_html_()[:6]
    assert styler._repr_latex_() is None
    with option_context('styler.render.repr', 'latex'):
        assert '\\begin{tabular}' in styler._repr_latex_()[:15]
        assert styler._repr_html_() is None