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
def test_use_locale(self):
    conv = locale.localeconv()
    sep = conv['thousands_sep']
    if not sep or conv['grouping'][-1:] in ([], [locale.CHAR_MAX]):
        pytest.skip('Locale does not apply grouping')
    with mpl.rc_context({'axes.formatter.use_locale': True}):
        tmp_form = mticker.ScalarFormatter()
        assert tmp_form.get_useLocale()
        tmp_form.create_dummy_axis()
        tmp_form.axis.set_data_interval(0, 10)
        tmp_form.set_locs([1, 2, 3])
        assert sep in tmp_form(1000000000.0)