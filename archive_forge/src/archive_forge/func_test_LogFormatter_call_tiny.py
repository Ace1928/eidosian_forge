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
@pytest.mark.parametrize('val', [1e-323, 2e-323, 1e-322, 1.1e-322])
def test_LogFormatter_call_tiny(self, val):
    temp_lf = mticker.LogFormatter()
    temp_lf.create_dummy_axis()
    temp_lf.axis.set_view_interval(1, 10)
    temp_lf(val)