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
@pytest.mark.parametrize('use_math_text', useMathText_data)
def test_useMathText(self, use_math_text):
    with mpl.rc_context({'axes.formatter.use_mathtext': use_math_text}):
        tmp_form = mticker.ScalarFormatter()
        assert use_math_text == tmp_form.get_useMathText()