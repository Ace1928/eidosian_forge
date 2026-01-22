from datetime import (
from importlib import reload
import string
import sys
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def rand_str(nchars: int) -> str:
    """
    Generate one random byte string.
    """
    RANDS_CHARS = np.array(list(string.ascii_letters + string.digits), dtype=(np.str_, 1))
    return ''.join(np.random.default_rng(2).choice(RANDS_CHARS, nchars))