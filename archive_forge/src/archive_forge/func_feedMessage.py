from twisted.cred import checkers, portal
from twisted.internet import defer, reactor
from twisted.protocols import sip
from twisted.trial import unittest
from zope.interface import implementer
def feedMessage(self, message):
    for c in message:
        self.parser.dataReceived(c)
    self.parser.dataDone()