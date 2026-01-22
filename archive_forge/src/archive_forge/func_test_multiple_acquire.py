import threading
import uuid
from os_win import exceptions
from os_win.tests.functional import test_base
from os_win.utils import processutils
def test_multiple_acquire(self):
    self._mutex.acquire(timeout_ms=0)
    self._mutex.acquire(timeout_ms=0)
    self._mutex.release()
    self._mutex.release()