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
@classmethod
def lookupProxyId(cls, pid):
    return cls.proxiedObjects[pid]