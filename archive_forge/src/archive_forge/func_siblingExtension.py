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
def siblingExtension(self, ext: OtherAnyStr) -> FilePath[OtherAnyStr]:
    """
        Attempt to return a path with my name, given the extension at C{ext}.

        @param ext: File-extension to search for.
        @type ext: L{bytes} or L{unicode}

        @return: The sibling path.
        @rtype: L{FilePath} with the same mode as the type of C{ext}.
        """
    ourPath = self._getPathAsSameTypeAs(ext)
    return self.clonePath(ourPath + ext)