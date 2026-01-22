import ctypes,sys
from ._arrayconstants import *
@property
def dim_strides(self):
    if self.strides:
        return self.strides[:self.ndim]
    return None