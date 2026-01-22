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
@contextlib.contextmanager
def create_destroy_green_thread(run_what, *args, **kwargs):
    t = eventlet.spawn(run_what, *args, **kwargs)
    try:
        yield
    finally:
        t.wait()