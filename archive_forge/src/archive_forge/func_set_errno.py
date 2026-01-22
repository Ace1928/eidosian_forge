import ctypes, ctypes.util, operator, sys
from . import model
def set_errno(self, value):
    ctypes.set_errno(value)