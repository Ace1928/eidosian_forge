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
def test_issue251_with_cleanup_disabled(self):
    greenlet._greenlet.enable_optional_cleanup(False)
    try:
        self._check_issue251()
    finally:
        greenlet._greenlet.enable_optional_cleanup(True)