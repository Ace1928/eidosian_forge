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
def test_polar_axes(self):
    """
        Polar axes have a different ticking logic.
        """
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_yscale('log')
    ax.set_ylim(1, 100)
    assert_array_equal(ax.get_yticks(), [10, 100, 1000])