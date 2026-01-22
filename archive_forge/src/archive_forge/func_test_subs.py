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
def test_subs(self):
    sym = mticker.SymmetricalLogLocator(base=10, linthresh=1, subs=[2.0, 4.0])
    sym.create_dummy_axis()
    sym.axis.set_view_interval(-10, 10)
    assert (sym() == [-20.0, -40.0, -2.0, -4.0, 0.0, 2.0, 4.0, 20.0, 40.0]).all()