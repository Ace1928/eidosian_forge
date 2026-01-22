import collections
import enum
import weakref
import astunparse
import gast
from tensorflow.python.autograph.pyct import anno
def visit_forward(self):
    self._visit_internal(_WalkMode.FORWARD)