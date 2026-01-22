from errno import EPERM
from socket import AF_INET, AF_INET6, IPPROTO_TCP, SOCK_STREAM, AddressFamily, gaierror
from types import FunctionType
from unicodedata import normalize
from unittest import skipIf
from zope.interface import implementer, providedBy, provider
from zope.interface.interface import InterfaceClass
from zope.interface.verify import verifyClass, verifyObject
from twisted import plugins
from twisted.internet import (
from twisted.internet.abstract import isIPv6Address
from twisted.internet.address import (
from twisted.internet.endpoints import StandardErrorBehavior
from twisted.internet.error import ConnectingCancelledError
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Factory, Protocol
from twisted.internet.stdio import PipeAddress
from twisted.internet.task import Clock
from twisted.internet.testing import (
from twisted.logger import ILogObserver, globalLogPublisher
from twisted.plugin import getPlugins
from twisted.protocols import basic, policies
from twisted.python import log
from twisted.python.compat import nativeString
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.modules import getModule
from twisted.python.systemd import ListenFDs
from twisted.test.iosim import connectableEndpoint, connectedServerAndClient
from twisted.trial import unittest
def replacingGlobals(function, **newGlobals):
    """
    Create a copy of the given function with the given globals substituted.

    The globals must already exist in the function's existing global scope.

    @param function: any function object.
    @type function: L{types.FunctionType}

    @param newGlobals: each keyword argument should be a global to set in the
        new function's returned scope.
    @type newGlobals: L{dict}

    @return: a new function, like C{function}, but with new global scope.
    """
    try:
        codeObject = function.func_code
        funcGlobals = function.func_globals
    except AttributeError:
        codeObject = function.__code__
        funcGlobals = function.__globals__
    for key in newGlobals:
        if key not in funcGlobals:
            raise TypeError('Name bound by replacingGlobals but not present in module: {}'.format(key))
    mergedGlobals = {}
    mergedGlobals.update(funcGlobals)
    mergedGlobals.update(newGlobals)
    newFunction = FunctionType(codeObject, mergedGlobals)
    mergedGlobals[function.__name__] = newFunction
    return newFunction