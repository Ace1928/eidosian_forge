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
@pytest.mark.parametrize('use_rcparam', [False, True])
@pytest.mark.parametrize('lim, ref', [((0, 1.39), [0.05, 0.1, 0.15, 0.25, 0.3, 0.35, 0.45, 0.5, 0.55, 0.65, 0.7, 0.75, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.25, 1.3, 1.35]), ((0, 0.139), [0.005, 0.01, 0.015, 0.025, 0.03, 0.035, 0.045, 0.05, 0.055, 0.065, 0.07, 0.075, 0.085, 0.09, 0.095, 0.105, 0.11, 0.115, 0.125, 0.13, 0.135])])
def test_number_of_minor_ticks_auto(self, lim, ref, use_rcparam):
    if use_rcparam:
        context = {'xtick.minor.ndivs': 'auto', 'ytick.minor.ndivs': 'auto'}
        kwargs = {}
    else:
        context = {}
        kwargs = {'n': 'auto'}
    with mpl.rc_context(context):
        fig, ax = plt.subplots()
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(**kwargs))
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(**kwargs))
        assert_almost_equal(ax.xaxis.get_ticklocs(minor=True), ref)
        assert_almost_equal(ax.yaxis.get_ticklocs(minor=True), ref)