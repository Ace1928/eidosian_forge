from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
import collections
import warnings
def get_value_type(self, name):
    """
        Retrieves the low-level data type for the given value.

        @type  name: str
        @param name: Registry value name.

        @rtype:  int
        @return: One of the following constants:
         - L{win32.REG_NONE} (0)
         - L{win32.REG_SZ} (1)
         - L{win32.REG_EXPAND_SZ} (2)
         - L{win32.REG_BINARY} (3)
         - L{win32.REG_DWORD} (4)
         - L{win32.REG_DWORD_BIG_ENDIAN} (5)
         - L{win32.REG_LINK} (6)
         - L{win32.REG_MULTI_SZ} (7)
         - L{win32.REG_RESOURCE_LIST} (8)
         - L{win32.REG_FULL_RESOURCE_DESCRIPTOR} (9)
         - L{win32.REG_RESOURCE_REQUIREMENTS_LIST} (10)
         - L{win32.REG_QWORD} (11)

        @raise KeyError: The specified value could not be found.
        """
    try:
        return win32.RegQueryValueEx(self.handle, name)[1]
    except WindowsError:
        e = sys.exc_info()[1]
        if e.winerror == win32.ERROR_FILE_NOT_FOUND:
            raise KeyError(name)
        raise