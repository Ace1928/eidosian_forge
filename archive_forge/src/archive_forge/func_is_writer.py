import collections
import contextlib
import threading
from fasteners import _utils
import six
def is_writer(self, check_pending=True):
    """Returns if the caller is the active writer or a pending writer."""
    me = self._current_thread()
    if self._writer == me:
        return True
    if check_pending:
        return me in self._pending_writers
    else:
        return False