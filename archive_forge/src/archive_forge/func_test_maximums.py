from textwrap import (
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
@pytest.mark.parametrize('rows', [True, False])
@pytest.mark.parametrize('cols', [True, False])
def test_maximums(styler_mi, rows, cols):
    result = styler_mi.to_html(max_rows=2 if rows else None, max_columns=2 if cols else None)
    assert '>5</td>' in result
    assert ('>8</td>' in result) is not rows
    assert ('>2</td>' in result) is not cols