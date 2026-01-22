import copy
from numpy.testing import (assert_almost_equal, assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import cos, sin
from scipy.optimize import basinhopping, OptimizeResult
from scipy.optimize._basinhopping import (
def myTakeStep2(x):
    """redo RandomDisplacement in function form without the attribute stepsize
    to make sure everything still works ok
    """
    s = 0.5
    x += np.random.uniform(-s, s, np.shape(x))
    return x