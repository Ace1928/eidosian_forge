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
@pytest.mark.parametrize('major_step, expected_nb_minordivisions', majorstep_minordivisions)
def test_number_of_minor_ticks(self, major_step, expected_nb_minordivisions):
    fig, ax = plt.subplots()
    xlims = (0, major_step)
    ax.set_xlim(*xlims)
    ax.set_xticks(xlims)
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    nb_minor_divisions = len(ax.xaxis.get_minorticklocs()) + 1
    assert nb_minor_divisions == expected_nb_minordivisions