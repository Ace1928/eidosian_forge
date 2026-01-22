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
def run_it():
    glets = []
    for _ in range(ITER):
        g = greenlet.greenlet(f)
        glets.append(g)
        g.switch()
    return glets