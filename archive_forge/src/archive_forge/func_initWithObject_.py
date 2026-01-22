import sys
import platform
import struct
from contextlib import contextmanager
from ctypes import *
from ctypes import util
from .cocoatypes import *
@DeallocationObserver.rawmethod('@@')
def initWithObject_(self, cmd, objc_ptr):
    self = send_super(self, 'init')
    self = self.value
    set_instance_variable(self, 'observed_object', objc_ptr, c_void_p)
    return self