import pytest
from numpy.testing import assert_allclose, assert_equal
import numpy as np
from math import pow
from scipy import optimize
TNC non-linear optimization.

    These tests are taken from Prof. K. Schittkowski's test examples
    for constrained non-linear programming.

    http://www.uni-bayreuth.de/departments/math/~kschittkowski/home.htm

    