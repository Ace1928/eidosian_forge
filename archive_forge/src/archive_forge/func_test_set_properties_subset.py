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
def test_set_properties_subset(self):
    df = DataFrame({'A': [0, 1]})
    result = df.style.set_properties(subset=IndexSlice[0, 'A'], color='white')._compute().ctx
    expected = {(0, 0): [('color', 'white')]}
    assert result == expected