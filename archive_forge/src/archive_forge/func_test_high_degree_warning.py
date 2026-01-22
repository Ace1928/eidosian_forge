import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def test_high_degree_warning(self):
    with pytest.warns(UserWarning, match='40 degrees provided,'):
        KroghInterpolator(np.arange(40), np.ones(40))