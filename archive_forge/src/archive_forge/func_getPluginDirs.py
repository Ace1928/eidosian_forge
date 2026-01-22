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
def getPluginDirs():
    warnings.warn('twisted.python.util.getPluginDirs is deprecated since Twisted 12.2.', DeprecationWarning, stacklevel=2)
    import twisted
    systemPlugins = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(twisted.__file__))), 'plugins')
    userPlugins = os.path.expanduser('~/TwistedPlugins')
    confPlugins = os.path.expanduser('~/.twisted')
    allPlugins = filter(os.path.isdir, [systemPlugins, userPlugins, confPlugins])
    return allPlugins