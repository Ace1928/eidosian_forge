import array
import pytest
import rpy2.robjects as robjects
def test_rclass_set_usingstring():
    x = robjects.r('1:3')
    old_class = x.rclass
    x.rclass = 'Foo'
    assert x.rclass[0] == 'Foo'