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
def test_set_use_offset_float(self):
    tmp_form = mticker.ScalarFormatter()
    tmp_form.set_useOffset(0.5)
    assert not tmp_form.get_useOffset()
    assert tmp_form.offset == 0.5