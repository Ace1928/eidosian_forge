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
def test_wide_values(self):
    lctr = mticker.AsinhLocator(linear_width=0.1, numticks=11, base=0)
    assert_almost_equal(lctr.tick_values(-100, 100), [-100, -20, -5, -1, -0.2, 0, 0.2, 1, 5, 20, 100])
    assert_almost_equal(lctr.tick_values(-1000, 1000), [-1000, -100, -20, -3, -0.4, 0, 0.4, 3, 20, 100, 1000])