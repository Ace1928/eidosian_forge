from __future__ import absolute_import
import cython
from .Transitions import TransitionMap
def is_accepting(self):
    return self.action is not None