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
@pytest.mark.parametrize('lims, expected_low_ticks', zip(ref_basic_limits, ref_basic_major_ticks))
def test_basic_major(self, lims, expected_low_ticks):
    """
        Create logit locator with huge number of major, and tests ticks.
        """
    expected_ticks = sorted([*expected_low_ticks, 0.5, *1 - expected_low_ticks])
    loc = mticker.LogitLocator(nbins=100)
    _LogitHelper.assert_almost_equal(loc.tick_values(*lims), expected_ticks)