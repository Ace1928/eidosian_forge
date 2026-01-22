from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gc
import sys
import time
import threading
from abc import ABCMeta, abstractmethod
import greenlet
from greenlet import greenlet as RawGreenlet
from . import TestCase
from .leakcheck import fails_leakcheck
def test_threaded_reparent(self):
    data = {}
    created_event = threading.Event()
    done_event = threading.Event()

    def run():
        data['g'] = RawGreenlet(lambda: None)
        created_event.set()
        done_event.wait(10)

    def blank():
        greenlet.getcurrent().parent.switch()
    thread = threading.Thread(target=run)
    thread.start()
    created_event.wait(10)
    g = RawGreenlet(blank)
    g.switch()
    with self.assertRaises(ValueError) as exc:
        g.parent = data['g']
    done_event.set()
    thread.join(10)
    self.assertEqual(str(exc.exception), 'parent cannot be on a different thread')