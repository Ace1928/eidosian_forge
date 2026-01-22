import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def check_strict_compare(self, k1, k2, mismatched_types):
    """True if on Python 3 and stricter comparison semantics are used."""
    if mismatched_types:
        for op in ('ge', 'gt', 'le', 'lt'):
            self.assertRaises(TypeError, getattr(operator, op), k1, k2)
        return True
    return False