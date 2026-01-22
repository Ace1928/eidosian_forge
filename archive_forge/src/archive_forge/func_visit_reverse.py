import collections
import enum
import weakref
import astunparse
import gast
from tensorflow.python.autograph.pyct import anno
def visit_reverse(self):
    self._visit_internal(_WalkMode.REVERSE)