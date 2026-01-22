import copy
import datetime
import decimal
import types
import warnings
from functools import reduce
from zope.interface import implementer
from incremental import Version
from twisted.persisted.crefutil import (
from twisted.python.compat import nativeString
from twisted.python.deprecate import deprecatedModuleAttribute
from twisted.python.reflect import namedAny, namedObject, qual
from twisted.spread.interfaces import IJellyable, IUnjellyable
def setUnjellyableForClassTree(module, baseClass, prefix=None):
    """
    Set all classes in a module derived from C{baseClass} as copiers for
    a corresponding remote class.

    When you have a hierarchy of Copyable (or Cacheable) classes on one
    side, and a mirror structure of Copied (or RemoteCache) classes on the
    other, use this to setUnjellyableForClass all your Copieds for the
    Copyables.

    Each copyTag (the "classname" argument to getTypeToCopyFor, and
    what the Copyable's getTypeToCopyFor returns) is formed from
    adding a prefix to the Copied's class name.  The prefix defaults
    to module.__name__.  If you wish the copy tag to consist of solely
    the classname, pass the empty string ''.

    @param module: a module object from which to pull the Copied classes.
        (passing sys.modules[__name__] might be useful)

    @param baseClass: the base class from which all your Copied classes derive.

    @param prefix: the string prefixed to classnames to form the
        unjellyableRegistry.
    """
    if prefix is None:
        prefix = module.__name__
    if prefix:
        prefix = '%s.' % prefix
    for name in dir(module):
        loaded = getattr(module, name)
        try:
            yes = issubclass(loaded, baseClass)
        except TypeError:
            "It's not a class."
        else:
            if yes:
                setUnjellyableForClass(f'{prefix}{name}', loaded)