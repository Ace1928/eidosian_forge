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
@pytest.mark.parametrize('sub', [['hi', 'aardvark'], np.zeros((2, 2))])
def test_bad_locator_subs(sub):
    ll = mticker.LogLocator()
    with pytest.raises(ValueError):
        ll.set_params(subs=sub)