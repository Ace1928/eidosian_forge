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
@pytest.mark.parametrize('N', (100, 253, 754))
def test_format_data_short(self, N):
    locs = np.linspace(0, 1, N)[1:-1]
    form = mticker.LogitFormatter()
    for x in locs:
        fx = form.format_data_short(x)
        if fx.startswith('1-'):
            x2 = 1 - float(fx[2:])
        else:
            x2 = float(fx)
        assert abs(x - x2) < 1 / N