import threading
import fasteners
from fasteners import test
@fasteners.write_locked
def i_am_write_locked(self, cb):
    cb(self._lock.owner)