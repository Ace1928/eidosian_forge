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
def deleteProxy(self, ref):
    if self.send is None:
        return
    with self.proxyLock:
        proxyId = self.proxies.pop(ref)
    try:
        self.send(request='del', opts=dict(proxyId=proxyId), callSync='off')
    except ClosedError:
        pass