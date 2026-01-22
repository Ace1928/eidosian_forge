import threading
from oslo_utils import timeutils
def countdown(self):
    """Decrements the internal counter due to an arrival."""
    with self._cond:
        self._count -= 1
        if self._count <= 0:
            self._cond.notify_all()