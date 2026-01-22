import os
import copy
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import scipy.spatial._qhull as qhull
from scipy.spatial import cKDTree as KDTree
from scipy.spatial import Voronoi
import itertools
def barycentric_transform(tr, x):
    r = tr[:, -1, :]
    Tinv = tr[:, :-1, :]
    return np.einsum('ijk,ik->ij', Tinv, x - r)