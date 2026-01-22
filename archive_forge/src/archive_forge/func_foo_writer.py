import contextlib
import io
import logging
import sys
import threading
import time
import unittest
from traits.api import HasTraits, Str, Int, Float, Any, Event
from traits.api import push_exception_handler, pop_exception_handler
def foo_writer(a, stop_event):
    while not stop_event.is_set():
        try:
            a.foo = True
        except Exception as e:
            a.exception = e