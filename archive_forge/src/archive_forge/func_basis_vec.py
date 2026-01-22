import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def basis_vec(axis):
    if axis == 'x':
        return [1, 0, 0]
    elif axis == 'y':
        return [0, 1, 0]
    elif axis == 'z':
        return [0, 0, 1]