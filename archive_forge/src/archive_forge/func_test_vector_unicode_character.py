import pytest
import sys
from rpy2 import robjects
from rpy2 import rinterface
import rpy2.rlike.container
import rpy2.robjects.conversion as conversion
def test_vector_unicode_character(self):
    l = [u'a', u'c', u'e']
    u = numpy.array(l, dtype='U')
    u_r = self.check_homogeneous(u, 'character', 'character')
    assert tuple(l) == tuple(u_r)