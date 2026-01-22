import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def is_writeable(self):
    """
        @rtype:  bool
        @return: C{True} if all pages in this region are writeable.
        """
    return self.has_content() and bool(self.Protect & self.WRITEABLE)