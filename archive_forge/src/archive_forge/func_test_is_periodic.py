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
def test_is_periodic(self):

    @periodics.periodic(0.5, enabled=False)
    def no_add_me():
        pass

    @periodics.periodic(0.5)
    def add_me():
        pass
    self.assertTrue(periodics.is_periodic(add_me))
    self.assertTrue(periodics.is_periodic(no_add_me))
    self.assertFalse(periodics.is_periodic(self.test_is_periodic))
    self.assertFalse(periodics.is_periodic(42))