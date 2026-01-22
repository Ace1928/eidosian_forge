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
def test_last_started_strategy(self):
    last_now = 3.2
    nows = [0, 2, 2, 3, -1]
    nows = list(reversed(nows))
    self._test_strategy('last_started', nows, last_now, 4.0)