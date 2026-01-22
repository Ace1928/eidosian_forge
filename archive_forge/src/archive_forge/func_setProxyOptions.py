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
def setProxyOptions(self, **kwds):
    """
        Set the default behavior options for object proxies.
        See ObjectProxy._setProxyOptions for more info.
        """
    with self.optsLock:
        self.proxyOptions.update(kwds)