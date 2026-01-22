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
def childSearchPreauth(self, *paths: OtherAnyStr) -> Optional[FilePath[OtherAnyStr]]:
    """
        Return my first existing child with a name in C{paths}.

        C{paths} is expected to be a list of *pre-secured* path fragments;
        in most cases this will be specified by a system administrator and not
        an arbitrary user.

        If no appropriately-named children exist, this will return L{None}.

        @return: L{None} or the child path.
        @rtype: L{None} or L{FilePath}
        """
    for child in paths:
        p = self._getPathAsSameTypeAs(child)
        jp = joinpath(p, child)
        if exists(jp):
            return self.clonePath(jp)
    return None