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
def test_render_trimming_mi():
    midx = MultiIndex.from_product([[1, 2], [1, 2, 3]])
    df = DataFrame(np.arange(36).reshape(6, 6), columns=midx, index=midx)
    with option_context('styler.render.max_elements', 4):
        ctx = df.style._translate(True, True)
    assert len(ctx['body'][0]) == 5
    assert {'attributes': 'rowspan="2"'}.items() <= ctx['body'][0][0].items()
    assert {'class': 'data row0 col_trim'}.items() <= ctx['body'][0][4].items()
    assert {'class': 'data row_trim col_trim'}.items() <= ctx['body'][2][4].items()
    assert len(ctx['body']) == 3