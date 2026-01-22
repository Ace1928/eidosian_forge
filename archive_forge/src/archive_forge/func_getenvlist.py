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
def getenvlist(self, encoding='utf-8'):
    """A wrapper for the pam_getenvlist function
        Returns:
          environment as python dictionary
        """
    if not self.handle:
        return PAM_SYSTEM_ERR
    env_list = self.pam_getenvlist(self.handle)
    env_count = 0
    pam_env_items = {}
    while True:
        try:
            item = env_list[env_count]
        except IndexError:
            break
        if not item:
            break
        env_item = item
        if sys.version_info >= (3,):
            env_item = env_item.decode(encoding)
        try:
            pam_key, pam_value = env_item.split('=', 1)
        except ValueError:
            pass
        else:
            pam_env_items[pam_key] = pam_value
        env_count += 1
    return pam_env_items