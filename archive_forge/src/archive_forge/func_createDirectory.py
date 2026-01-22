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
def createDirectory(self) -> None:
    """
        Create the directory the L{FilePath} refers to.

        @see: L{makedirs}

        @raise OSError: If the directory cannot be created.
        """
    os.mkdir(self.path)