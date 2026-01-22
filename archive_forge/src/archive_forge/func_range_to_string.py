from __future__ import absolute_import
import cython
from .Transitions import TransitionMap
def range_to_string(self, range_tuple):
    c1, c2 = range_tuple
    if c1 == c2:
        return repr(c1)
    else:
        return '%s..%s' % (repr(c1), repr(c2))