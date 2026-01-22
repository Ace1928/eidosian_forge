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
def restat(self, reraise: bool=True) -> None:
    """
        Re-calculate cached effects of 'stat'.  To refresh information on this
        path after you know the filesystem may have changed, call this method.

        @param reraise: a boolean.  If true, re-raise exceptions from
            L{os.stat}; otherwise, mark this path as not existing, and remove
            any cached stat information.

        @raise Exception: If C{reraise} is C{True} and an exception occurs
            while reloading metadata.
        """
    try:
        self._statinfo = stat(self.path)
    except OSError:
        self._statinfo = None
        if reraise:
            raise