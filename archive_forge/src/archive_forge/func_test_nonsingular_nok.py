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
@pytest.mark.parametrize('okval', acceptable_vmin_vmax)
def test_nonsingular_nok(self, okval):
    """
        Create logit locator, and test the nonsingular method for non
        acceptable value
        """
    loc = mticker.LogitLocator()
    vmin, vmax = (-1, okval)
    vmin2, vmax2 = loc.nonsingular(vmin, vmax)
    assert vmax2 == vmax
    assert 0 < vmin2 < vmax2
    vmin, vmax = (okval, 2)
    vmin2, vmax2 = loc.nonsingular(vmin, vmax)
    assert vmin2 == vmin
    assert vmin2 < vmax2 < 1