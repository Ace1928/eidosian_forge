import pytest
import rpy2.robjects as robjects
import array
def test_svd():
    m = robjects.r.matrix(robjects.IntVector((1, -1, -1, 1)), nrow=2)
    res = m.svd()
    for i, val in enumerate(res.rx2('d')):
        assert almost_equal((2, 0)[i], val)