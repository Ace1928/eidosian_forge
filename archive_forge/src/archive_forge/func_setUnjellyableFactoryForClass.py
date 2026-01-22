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
def setUnjellyableFactoryForClass(classname, copyFactory):
    """
    Set the factory to construct a remote instance of a type::

      jellier.setUnjellyableFactoryForClass('module.package.Class', MyFactory)

    Call this at the module level immediately after its class definition.
    C{copyFactory} should return an instance or subclass of
    L{RemoteCopy<pb.RemoteCopy>}.

    Similar to L{setUnjellyableForClass} except it uses a factory instead
    of creating an instance.
    """
    global unjellyableFactoryRegistry
    classname = _maybeClass(classname)
    unjellyableFactoryRegistry[classname] = copyFactory
    globalSecurity.allowTypes(classname)