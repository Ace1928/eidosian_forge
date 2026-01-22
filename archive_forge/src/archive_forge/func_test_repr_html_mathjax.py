import contextlib
import copy
import re
from textwrap import dedent
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.formats.style import (  # isort:skip
from pandas.io.formats.style_render import (
def test_repr_html_mathjax(self, styler):
    assert 'tex2jax_ignore' not in styler._repr_html_()
    with option_context('styler.html.mathjax', False):
        assert 'tex2jax_ignore' in styler._repr_html_()