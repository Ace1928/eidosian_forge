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
@pytest.mark.parametrize('min_exponent, value, expected', test_data)
def test_min_exponent(self, min_exponent, value, expected):
    with mpl.rc_context({'axes.formatter.min_exponent': min_exponent}):
        assert self.fmt(value) == expected