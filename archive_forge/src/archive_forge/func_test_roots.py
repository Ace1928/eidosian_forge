import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def test_roots(self):
    p = pchip([0, 1], [-1, 1])
    r = p.roots()
    assert_allclose(r, 0.5)