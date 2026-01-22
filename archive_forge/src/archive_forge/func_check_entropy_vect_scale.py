import pickle
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose, assert_equal
from pytest import raises as assert_raises
import numpy.ma.testutils as ma_npt
from scipy._lib._util import (
from scipy import stats
def check_entropy_vect_scale(distfn, arg):
    sc = np.asarray([[1, 2], [3, 4]])
    v_ent = distfn.entropy(*arg, scale=sc)
    s_ent = [distfn.entropy(*arg, scale=s) for s in sc.ravel()]
    s_ent = np.asarray(s_ent).reshape(v_ent.shape)
    assert_allclose(v_ent, s_ent, atol=1e-14)
    sc = [1, 2, -3]
    v_ent = distfn.entropy(*arg, scale=sc)
    s_ent = [distfn.entropy(*arg, scale=s) for s in sc]
    s_ent = np.asarray(s_ent).reshape(v_ent.shape)
    assert_allclose(v_ent, s_ent, atol=1e-14)