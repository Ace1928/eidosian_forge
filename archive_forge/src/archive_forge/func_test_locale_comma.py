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
def test_locale_comma():
    proc = mpl.testing.subprocess_run_helper(_impl_locale_comma, timeout=60, extra_env={'MPLBACKEND': 'Agg'})
    skip_msg = next((line[len('SKIP:'):].strip() for line in proc.stdout.splitlines() if line.startswith('SKIP:')), '')
    if skip_msg:
        pytest.skip(skip_msg)