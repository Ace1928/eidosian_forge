import collections
import functools
import threading
import time
from taskflow import test
from taskflow.utils import threading_utils as tu
def test_alive_thread(self):
    death = threading.Event()
    t = tu.daemon_thread(_spinner, death)
    self.assertFalse(tu.is_alive(t))
    t.start()
    self.assertTrue(tu.is_alive(t))
    death.set()
    t.join()
    self.assertFalse(tu.is_alive(t))