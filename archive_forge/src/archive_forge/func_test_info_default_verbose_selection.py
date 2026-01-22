from io import StringIO
import re
from string import ascii_uppercase as uppercase
import sys
import textwrap
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('num_columns, max_info_columns, verbose', [(10, 100, True), (10, 11, True), (10, 10, True), (10, 9, False), (10, 1, False)])
def test_info_default_verbose_selection(num_columns, max_info_columns, verbose):
    frame = DataFrame(np.random.randn(5, num_columns))
    with option_context('display.max_info_columns', max_info_columns):
        io_default = StringIO()
        frame.info(buf=io_default)
        result = io_default.getvalue()
        io_explicit = StringIO()
        frame.info(buf=io_explicit, verbose=verbose)
        expected = io_explicit.getvalue()
        assert result == expected