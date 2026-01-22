import ctypes
import itertools
import numpy as np
from . import QT_LIB, QtCore, QtGui, compat
def drawargs(self):
    if self.use_ptr_to_array:
        if self._capa > 0:
            ptr = compat.wrapinstance(self._ndarray.ctypes.data, self._Klass)
        else:
            ptr = None
        return (ptr, self._size)
    else:
        return (self.instances(),)