import numpy as np
import scipy.sparse as ssp
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
def test_scipy_sparse():
    foo = ssp.lil_matrix(np.eye(3, k=1))
    goo = foo.getrowview(0)
    goo[goo.nonzero()] = 0
    assert foo[0, 1] == 0