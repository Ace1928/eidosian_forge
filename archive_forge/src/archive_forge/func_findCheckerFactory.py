import sys
from typing import Optional, Sequence, Type
from zope.interface import Attribute, Interface
from twisted.plugin import getPlugins
from twisted.python import usage
def findCheckerFactory(authType):
    """
    Find the first checker factory that supports the given authType.
    """
    for factory in findCheckerFactories():
        if factory.authType == authType:
            return factory
    raise InvalidAuthType(authType)