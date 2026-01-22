from __future__ import annotations
import errno
import os
import sys
import warnings
from typing import AnyStr
from collections import OrderedDict
from typing import (
from incremental import Version
from twisted.python.deprecate import deprecatedModuleAttribute
def gidFromString(gidString):
    """
    Convert a group identifier, as a string, into an integer GID.

    @type gidString: C{str}
    @param gidString: A string giving the base-ten representation of a GID or
        the name of a group which can be converted to a GID via L{grp.getgrnam}.

    @rtype: C{int}
    @return: The integer GID corresponding to the given string.

    @raise ValueError: If the group name is supplied and L{grp} is not
        available.
    """
    try:
        return int(gidString)
    except ValueError:
        if grp is None:
            raise
        return grp.getgrnam(gidString)[2]