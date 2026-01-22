import threading
import fasteners
from fasteners import test
def i_am_not_locked(self, cb):
    cb(self._lock.owner)