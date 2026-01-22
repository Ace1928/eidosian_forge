import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def get_tid(self):
    """
        @rtype:  int
        @return: Thread global ID.
        """
    return GetThreadId(self.value)