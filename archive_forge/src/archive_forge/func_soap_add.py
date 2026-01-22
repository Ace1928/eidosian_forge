from unittest import skipIf
from twisted.internet import defer, reactor
from twisted.trial.unittest import TestCase
from twisted.web import error, server
def soap_add(self, a, b):
    return a + b