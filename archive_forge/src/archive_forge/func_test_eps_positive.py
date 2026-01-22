import warnings
import platform
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.tests._locales import CommaDecimalPointLocale
def test_eps_positive():
    assert np.finfo(np.longdouble).eps > 0.0