import collections
import contextlib
import threading
from fasteners import _utils
import six
def is_reader(self):
    """Returns if the caller is one of the readers."""
    me = self._current_thread()
    return me in self._readers