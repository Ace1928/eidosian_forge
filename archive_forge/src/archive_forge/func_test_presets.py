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
def test_presets(self):
    loc = mticker.LinearLocator(presets={(1, 2): [1, 1.25, 1.75], (0, 2): [0.5, 1.5]})
    assert loc.tick_values(1, 2) == [1, 1.25, 1.75]
    assert loc.tick_values(2, 1) == [1, 1.25, 1.75]
    assert loc.tick_values(0, 2) == [0.5, 1.5]
    assert loc.tick_values(0.0, 2.0) == [0.5, 1.5]
    assert (loc.tick_values(0, 1) == np.linspace(0, 1, 11)).all()