import datetime
import decimal
from typing import ClassVar, Dict, Type, TypeVar
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
from twisted.internet import address, defer, error, interfaces, protocol, reactor
from twisted.internet.testing import StringTransport
from twisted.protocols import amp
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.test import iosim
from twisted.trial.unittest import TestCase
def cmdHello(self, hello, From, optional=None, Print=None, mixedCase=None, dash_arg=None, underscore_arg=None):
    assert From == self.transport.getPeer()
    if hello == THING_I_DONT_UNDERSTAND:
        raise ThingIDontUnderstandError()
    if hello.startswith(b'fuck'):
        raise UnfriendlyGreeting("Don't be a dick.")
    if hello == b'die':
        raise DeathThreat('aieeeeeeeee')
    result = dict(hello=hello)
    if Print is not None:
        result.update(dict(Print=Print))
    self.greeted = True
    return result