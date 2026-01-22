from ctypes import *
from ctypes import util
from .runtime import send_message, ObjCInstance
from .cocoatypes import *
def get_NSString(string):
    """Autoreleased version of CFSTR"""
    return ObjCInstance(c_void_p(CFSTR(string))).autorelease()