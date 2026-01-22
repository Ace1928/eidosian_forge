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
def misc_setenv(self, name, value, readonly, encoding='utf-8'):
    """A wrapper for the pam_misc_setenv function
        Args:
          name: key name of the environment variable
          value: the value of the environment variable
        Returns:
          Linux-PAM return value as int
        """
    if not self.handle or not hasattr(self, 'pam_misc_setenv'):
        return PAM_SYSTEM_ERR
    return self.pam_misc_setenv(self.handle, name.encode(encoding), value.encode(encoding), readonly)