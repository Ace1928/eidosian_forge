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
def send_exception(g, exc):

    def crasher(exc):
        raise exc
    g1 = RawGreenlet(crasher, parent=g)
    g1.switch(exc)