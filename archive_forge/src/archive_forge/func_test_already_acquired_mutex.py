import threading
import uuid
from os_win import exceptions
from os_win.tests.functional import test_base
from os_win.utils import processutils
def test_already_acquired_mutex(self):
    thread, stop_event = self.acquire_mutex_in_separate_thread(self._mutex)
    self.assertFalse(self._mutex.acquire(timeout_ms=0))
    stop_event.set()
    self.assertTrue(self._mutex.acquire(timeout_ms=2000))