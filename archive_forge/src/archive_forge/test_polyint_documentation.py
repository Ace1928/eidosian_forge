import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
Check that spline coefficients satisfy the continuity and boundary
        conditions.