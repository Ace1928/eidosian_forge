import array
import pytest
import rpy2.robjects as robjects
def test_rclass_str():
    s = str(robjects.r)
    assert isinstance(s, str)