from __future__ import print_function, absolute_import, division
import sys
import gc
import time
import weakref
import threading
import greenlet
from . import TestCase
from .leakcheck import fails_leakcheck
from .leakcheck import ignores_leakcheck
from .leakcheck import RUNNING_ON_MANYLINUX
@ignores_leakcheck
def test_untracked_memory_doesnt_increase(self):
    self._only_test_some_versions()

    def f():
        return 1
    ITER = 10000

    def run_it():
        for _ in range(ITER):
            greenlet.greenlet(f).switch()
    for _ in range(3):
        run_it()
    uss_before = self.get_process_uss()
    for count in range(self.UNTRACK_ATTEMPTS):
        uss_before = max(uss_before, self.get_process_uss())
        run_it()
        uss_after = self.get_process_uss()
        if uss_after <= uss_before and count > 1:
            break
    self.assertLessEqual(uss_after, uss_before)