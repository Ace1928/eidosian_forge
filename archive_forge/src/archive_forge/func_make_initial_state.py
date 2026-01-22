from __future__ import absolute_import
import cython
from .Transitions import TransitionMap
def make_initial_state(self, name, state):
    self.initial_states[name] = state