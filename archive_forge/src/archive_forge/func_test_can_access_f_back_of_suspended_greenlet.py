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
def test_can_access_f_back_of_suspended_greenlet(self):
    main = greenlet.getcurrent()

    def outer():
        inner()

    def inner():
        main.switch(sys._getframe(0))
    hub = RawGreenlet(outer)
    hub.switch()
    unrelated = RawGreenlet(lambda: None)
    unrelated.switch()
    self.assertIsNotNone(hub.gr_frame)
    self.assertEqual(hub.gr_frame.f_code.co_name, 'inner')
    self.assertIsNotNone(hub.gr_frame.f_back)
    self.assertEqual(hub.gr_frame.f_back.f_code.co_name, 'outer')
    self.assertIsNone(hub.gr_frame.f_back.f_back)