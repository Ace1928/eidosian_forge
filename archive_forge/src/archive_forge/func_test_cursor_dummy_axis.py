from contextlib import nullcontext
import itertools
import locale
import logging
import re
from packaging.version import parse as parse_version
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
@pytest.mark.parametrize('data, expected', cursor_data)
def test_cursor_dummy_axis(self, data, expected):
    sf = mticker.ScalarFormatter()
    sf.create_dummy_axis()
    sf.axis.set_view_interval(0, 10)
    fmt = sf.format_data_short
    assert fmt(data) == expected
    assert sf.axis.get_tick_space() == 9
    assert sf.axis.get_minpos() == 0