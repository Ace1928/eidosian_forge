import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def has_content(self):
    """
        @rtype:  bool
        @return: C{True} if the memory in this region has any data in it.
        """
    return self.is_commited() and (not bool(self.Protect & (PAGE_GUARD | PAGE_NOACCESS)))