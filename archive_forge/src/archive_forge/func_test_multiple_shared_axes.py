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
def test_multiple_shared_axes(self):
    rng = np.random.default_rng(19680801)
    dummy_data = [rng.normal(size=100), [], []]
    fig, axes = plt.subplots(len(dummy_data), sharex=True, sharey=True)
    for ax, data in zip(axes.flatten(), dummy_data):
        ax.hist(data, bins=10)
        ax.set_yscale('log', nonpositive='clip')
    for ax in axes.flatten():
        assert all(ax.get_yticks() == axes[0].get_yticks())
        assert ax.get_ylim() == axes[0].get_ylim()