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
def test_get_stack_with_nested_c_calls(self):
    from functools import partial
    from . import _test_extension_cpp

    def recurse(v):
        if v > 0:
            return v * _test_extension_cpp.test_call(partial(recurse, v - 1))
        return greenlet.getcurrent().parent.switch()
    gr = RawGreenlet(recurse)
    gr.switch(5)
    frame = gr.gr_frame
    for i in range(5):
        self.assertEqual(frame.f_locals['v'], i)
        frame = frame.f_back
    self.assertEqual(frame.f_locals['v'], 5)
    self.assertIsNone(frame.f_back)
    self.assertEqual(gr.switch(10), 1200)