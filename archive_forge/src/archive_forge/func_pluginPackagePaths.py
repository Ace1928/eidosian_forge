import os
import pickle
import sys
import types
from typing import Iterable, Optional, Type, TypeVar
from zope.interface import Interface, providedBy
from twisted.python import log
from twisted.python.components import getAdapterFactory
from twisted.python.modules import getModule
from twisted.python.reflect import namedAny
def pluginPackagePaths(name):
    """
    Return a list of additional directories which should be searched for
    modules to be included as part of the named plugin package.

    @type name: C{str}
    @param name: The fully-qualified Python name of a plugin package, eg
        C{'twisted.plugins'}.

    @rtype: C{list} of C{str}
    @return: The absolute paths to other directories which may contain plugin
        modules for the named plugin package.
    """
    package = name.split('.')
    return [os.path.abspath(os.path.join(x, *package)) for x in sys.path if not os.path.exists(os.path.join(x, *package + ['__init__.py']))]