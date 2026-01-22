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
def getStatusChangeTime(self) -> float:
    """
        Retrieve the time of the last status change for this file.

        @return: a number of seconds from the epoch.
        @rtype: L{float}
        """
    st = self._statinfo
    if not st:
        self.restat()
        st = self._statinfo
    assert st is not None
    return float(st.st_ctime)