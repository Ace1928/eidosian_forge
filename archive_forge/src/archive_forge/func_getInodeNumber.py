from __future__ import annotations
import base64
import errno
import os
import sys
from os import listdir, stat, utime
from os.path import (
from stat import (
from typing import (
from zope.interface import Attribute, Interface, implementer
from typing_extensions import Literal
from twisted.python.compat import cmp, comparable
from twisted.python.runtime import platform
from twisted.python.util import FancyEqMixin
from twisted.python.win32 import (
def getInodeNumber(self) -> int:
    """
        Retrieve the file serial number, also called inode number, which
        distinguishes this file from all other files on the same device.

        @raise NotImplementedError: if the platform is Windows, since the
            inode number would be a dummy value for all files in Windows
        @return: a number representing the file serial number
        @rtype: L{int}
        @since: 11.0
        """
    if platform.isWindows():
        raise NotImplementedError
    st = self._statinfo
    if not st:
        self.restat()
        st = self._statinfo
    assert st is not None
    return st.st_ino