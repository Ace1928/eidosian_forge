import ctypes, ctypes.util, operator, sys
from . import model
def load_function(self, BType, name):
    c_func = getattr(self.cdll, name)
    funcobj = BType._from_ctypes(c_func)
    funcobj._name = name
    return funcobj