import os
import six
import sys
from ctypes import cdll
from ctypes import CFUNCTYPE
from ctypes import CDLL
from ctypes import POINTER
from ctypes import Structure
from ctypes import byref
from ctypes import cast
from ctypes import sizeof
from ctypes import py_object
from ctypes import c_char
from ctypes import c_char_p
from ctypes import c_int
from ctypes import c_size_t
from ctypes import c_void_p
from ctypes import memmove
from ctypes.util import find_library
from typing import Union
def open_session(self, encoding='utf-8'):
    """Call pam_open_session as required by the pam_api
        Returns:
          Linux-PAM return value as int
        """
    if not self.handle:
        return PAM_SYSTEM_ERR
    retval = self.pam_open_session(self.handle, 0)
    self.code = retval
    self.reason = self.pam_strerror(self.handle, retval)
    if sys.version_info >= (3,):
        self.reason = self.reason.decode(encoding)
    return retval