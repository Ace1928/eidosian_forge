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
def getObjValue(self, obj, **kwds):
    return self.send(request='getObjValue', opts=dict(obj=obj), **kwds)