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
@pytest.mark.parametrize('method, lims, cases', lims_minor_major)
def test_minor_vs_major(self, method, lims, cases):
    """
        Test minor/major displays.
        """
    if method:
        min_loc = mticker.LogitLocator(minor=True)
        ticks = min_loc.tick_values(*lims)
    else:
        ticks = np.array(lims)
    min_form = mticker.LogitFormatter(minor=True)
    for threshold, has_minor in cases:
        min_form.set_minor_threshold(threshold)
        formatted = min_form.format_ticks(ticks)
        labelled = [f for f in formatted if len(f) > 0]
        if has_minor:
            assert len(labelled) > 0, (threshold, has_minor)
        else:
            assert len(labelled) == 0, (threshold, has_minor)