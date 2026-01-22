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
@pytest.mark.parametrize('lims', ref_basic_limits + ref_maxn_limits)
def test_nbins_major(self, lims):
    """
        Assert logit locator for respecting nbins param.
        """
    basic_needed = int(-np.floor(np.log10(lims[0]))) * 2 + 1
    loc = mticker.LogitLocator(nbins=100)
    for nbins in range(basic_needed, 2, -1):
        loc.set_params(nbins=nbins)
        assert len(loc.tick_values(*lims)) <= nbins + 2