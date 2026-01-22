import sys
import os
import ctypes
import optparse
from winappdbg import win32
from winappdbg import compat
@classproperty
def pageSize(cls):
    """
        Try to get the pageSize value on runtime.
        """
    try:
        try:
            pageSize = win32.GetSystemInfo().dwPageSize
        except WindowsError:
            pageSize = 4096
    except NameError:
        pageSize = 4096
    cls.pageSize = pageSize
    return pageSize