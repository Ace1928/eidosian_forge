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
def test_chaining_table_styles(self):
    df = DataFrame(data=[[0, 1], [1, 2]], columns=['A', 'B'])
    styler = df.style.set_table_styles([{'selector': '', 'props': [('background-color', 'yellow')]}]).set_table_styles([{'selector': '.col0', 'props': [('background-color', 'blue')]}], overwrite=False)
    assert len(styler.table_styles) == 2