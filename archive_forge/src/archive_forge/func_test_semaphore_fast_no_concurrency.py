import queue
from threading import Thread
from timeit import default_timer as timer
from unittest import mock
import testtools
from keystoneauth1 import _fair_semaphore
def test_semaphore_fast_no_concurrency(self):
    self._concurrency_core(None, 0.0)