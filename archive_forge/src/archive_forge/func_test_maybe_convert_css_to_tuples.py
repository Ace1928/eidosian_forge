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
def test_maybe_convert_css_to_tuples(self):
    expected = [('a', 'b'), ('c', 'd e')]
    assert maybe_convert_css_to_tuples('a:b;c:d e;') == expected
    assert maybe_convert_css_to_tuples('a: b ;c:  d e  ') == expected
    expected = []
    assert maybe_convert_css_to_tuples('') == expected