import pytest
import rpy2.robjects as robjects
import array
def test_crossprod():
    m = robjects.r.matrix(robjects.IntVector(range(4)), nrow=2)
    mcp = m.crossprod(m)
    for i, val in enumerate((1.0, 3.0, 3.0, 13.0)):
        assert mcp[i] == val