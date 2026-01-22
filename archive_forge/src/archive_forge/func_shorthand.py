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
def shorthand(self) -> str:
    """
        Returns a short string representing the permission bits.  Looks like
        what is printed by command line utilities such as 'ls -l'
        (e.g. 'rwx-wx--x')

        @return: The shorthand string.
        @rtype: L{str}
        """
    return ''.join([x.shorthand() for x in (self.user, self.group, self.other)])