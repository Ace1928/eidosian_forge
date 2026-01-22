import os
import sys
import time
from itertools import zip_longest
import numpy as np
from numpy.testing import assert_
import pytest
from scipy.special._testutils import assert_func_equal
def nonfunctional_tooslow(func):
    return pytest.mark.skip(reason='    Test not yet functional (too slow), needs more work.')(func)