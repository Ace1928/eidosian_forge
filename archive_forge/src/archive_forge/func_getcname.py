import ctypes, ctypes.util, operator, sys
from . import model
def getcname(self, BType, replace_with):
    return BType._get_c_name(replace_with)