import collections
import weakref
from tensorflow.python.util import object_identity
def mark_as(self, value):
    may_affect_upstream = value != self._in_cached_state
    self._in_cached_state = value
    return may_affect_upstream