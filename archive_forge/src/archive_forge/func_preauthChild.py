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
def preauthChild(self, path: OtherAnyStr) -> FilePath[OtherAnyStr]:
    """
        Use me if C{path} might have slashes in it, but you know they're safe.

        @param path: A relative path (ie, a path not starting with C{"/"})
            which will be interpreted as a child or descendant of this path.
        @type path: L{bytes} or L{unicode}

        @return: The child path.
        @rtype: L{FilePath} with a mode equal to the type of C{path}.
        """
    ourPath = self._getPathAsSameTypeAs(path)
    newpath = abspath(joinpath(ourPath, normpath(path)))
    if not newpath.startswith(ourPath):
        raise InsecurePath(f'{newpath!r} is not a child of {ourPath!r}')
    return self.clonePath(newpath)