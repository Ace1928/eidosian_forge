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
def test_dealloc_other_thread(self):
    seen = []
    someref = []
    bg_glet_created_running_and_no_longer_ref_in_bg = threading.Event()
    fg_ref_released = threading.Event()
    bg_should_be_clear = threading.Event()
    ok_to_exit_bg_thread = threading.Event()

    def f():
        g1 = RawGreenlet(fmain)
        g1.switch(seen)
        someref.append(g1)
        del g1
        gc.collect()
        bg_glet_created_running_and_no_longer_ref_in_bg.set()
        fg_ref_released.wait(3)
        RawGreenlet()
        bg_should_be_clear.set()
        ok_to_exit_bg_thread.wait(3)
        RawGreenlet()
    t = threading.Thread(target=f)
    t.start()
    bg_glet_created_running_and_no_longer_ref_in_bg.wait(10)
    self.assertEqual(seen, [])
    self.assertEqual(len(someref), 1)
    del someref[:]
    gc.collect()
    self.assertEqual(seen, [])
    fg_ref_released.set()
    bg_should_be_clear.wait(3)
    try:
        self.assertEqual(seen, [greenlet.GreenletExit])
    finally:
        ok_to_exit_bg_thread.set()
        t.join(10)
        del seen[:]
        del someref[:]