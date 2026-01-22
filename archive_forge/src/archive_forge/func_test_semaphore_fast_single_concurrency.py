import queue
from threading import Thread
from timeit import default_timer as timer
from unittest import mock
import testtools
from keystoneauth1 import _fair_semaphore
def test_semaphore_fast_single_concurrency(self):
    self._concurrency_core(1, 0.0)