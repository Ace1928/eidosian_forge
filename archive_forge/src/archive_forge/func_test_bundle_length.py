import collections
import functools
import threading
import time
from taskflow import test
from taskflow.utils import threading_utils as tu
def test_bundle_length(self):
    self.assertEqual(0, len(self.bundle))
    for i in range(0, self.thread_count):
        self.bundle.bind(lambda: tu.daemon_thread(_spinner, self.death))
        self.assertEqual(1, self.bundle.start())
        self.assertEqual(i + 1, len(self.bundle))
    self.death.set()
    self.assertEqual(self.thread_count, self.bundle.stop())
    self.assertEqual(self.thread_count, len(self.bundle))