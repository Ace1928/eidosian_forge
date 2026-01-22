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
@pytest.mark.parametrize('numticks', [1, 2, 3, 9])
@mpl.style.context('default')
def test_small_range_loglocator(numticks):
    ll = mticker.LogLocator()
    ll.set_params(numticks=numticks)
    for top in [5, 7, 9, 11, 15, 50, 100, 1000]:
        ticks = ll.tick_values(0.5, top)
        assert (np.diff(np.log10(ll.tick_values(6, 150))) == 1).all()