import pytest
import rpy2.robjects as robjects
import array
def test_dim():
    m = robjects.r.matrix(1, nrow=5, ncol=3)
    a = robjects.vectors.FloatArray(m)
    d = a.dim
    assert len(d) == 2
    assert d[0] == 5
    assert d[1] == 3