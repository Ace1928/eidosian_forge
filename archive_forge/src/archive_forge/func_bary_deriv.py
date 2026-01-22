import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def bary_deriv(x, y, axis=0):
    return BarycentricInterpolator(x, y, axis).derivative