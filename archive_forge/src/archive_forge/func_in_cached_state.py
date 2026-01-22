import collections
import weakref
from tensorflow.python.util import object_identity
@property
def in_cached_state(self):
    return self._in_cached_state