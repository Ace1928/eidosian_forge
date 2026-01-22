import os
import sys
import threading
import time
import traceback
import warnings
import weakref
import builtins
import pickle
import numpy as np
from ..util import cprint
def registerProxy(self, proxy):
    with self.proxyLock:
        ref = weakref.ref(proxy, self.deleteProxy)
        self.proxies[ref] = proxy._proxyId