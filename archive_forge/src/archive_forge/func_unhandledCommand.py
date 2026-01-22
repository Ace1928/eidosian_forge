from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def unhandledCommand(self, command, data):
    """
        Record the given command in C{self.events}.
        """
    self.events.append(('command', command, data))