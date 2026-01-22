import contextlib
import functools
import threading
import time
from unittest import mock
import eventlet
from eventlet.green import threading as green_threading
import testscenarios
import futurist
from futurist import periodics
from futurist.tests import base
def test_aligned_strategy(self):
    last_now = 5.5
    nows = [0, 2, 2, 5, -1]
    nows = list(reversed(nows))
    self._test_strategy('aligned_last_finished', nows, last_now, 6.0)