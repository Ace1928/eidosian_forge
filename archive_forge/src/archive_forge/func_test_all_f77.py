import pytest
from numpy import array
from . import util
@pytest.mark.slow
@pytest.mark.parametrize('name', 't0,t1,t2,t4,s0,s1,s2,s4'.split(','))
def test_all_f77(self, name):
    self.check_function(getattr(self.module, name))