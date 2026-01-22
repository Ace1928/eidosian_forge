import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def spl_interp(x, y, axis):
    return make_interp_spline(x, y, axis=axis)