import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_, assert_warns
import pytest
from pytest import raises as assert_raises
import scipy.cluster.hierarchy
from scipy.cluster.hierarchy import (
from scipy.spatial.distance import pdist
from scipy.cluster._hierarchy import Heap
from scipy.conftest import (
from scipy._lib._array_api import xp_assert_close
from . import hierarchy_test_data
def check_fclusterdata(self, t, criterion, xp):
    expectedT = xp.asarray(getattr(hierarchy_test_data, 'fcluster_' + criterion)[t])
    X = xp.asarray(hierarchy_test_data.Q_X)
    T = fclusterdata(X, criterion=criterion, t=t)
    assert_(is_isomorphic(T, expectedT))