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
def test_brokenReturnValue(self):
    """
        It can be very confusing if you write some code which responds to a
        command, but gets the return value wrong.  Most commonly you end up
        returning None instead of a dictionary.

        Verify that if that happens, the framework logs a useful error.
        """
    L = []
    SimpleSymmetricCommandProtocol().dispatchCommand(amp.AmpBox(_command=BrokenReturn.commandName)).addErrback(L.append)
    L[0].trap(amp.BadLocalReturn)
    self.failUnlessIn('None', repr(L[0].value))