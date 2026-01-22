import array
import pytest
import rpy2.robjects as robjects
def test_rclass_set():
    x = robjects.r('1:3')
    old_class = x.rclass
    x.rclass = robjects.StrVector(('Foo',)) + x.rclass
    assert x.rclass[0] == 'Foo'
    assert old_class[0] == x.rclass[1]