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
def test_one_half(self):
    """
        Test the parameter one_half
        """
    form = mticker.LogitFormatter()
    assert '\\frac{1}{2}' in form(1 / 2)
    form.set_one_half('1/2')
    assert '1/2' in form(1 / 2)
    form.set_one_half('one half')
    assert 'one half' in form(1 / 2)