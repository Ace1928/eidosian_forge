from ctypes import *
from ctypes import util
from .runtime import send_message, ObjCInstance
from .cocoatypes import *
def cfarray_to_list(cfarray):
    """Convert CFArray to python list."""
    count = cf.CFArrayGetCount(cfarray)
    return [cftype_to_value(c_void_p(cf.CFArrayGetValueAtIndex(cfarray, i))) for i in range(count)]