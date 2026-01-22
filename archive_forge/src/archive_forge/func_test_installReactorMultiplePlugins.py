import copy
import os
import pickle
from io import StringIO
from unittest import skipIf
from twisted.application import app, internet, reactors, service
from twisted.application.internet import backoffPolicy
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet.testing import MemoryReactor
from twisted.persisted import sob
from twisted.plugins import twisted_reactors
from twisted.protocols import basic, wire
from twisted.python import usage
from twisted.python.runtime import platformType
from twisted.python.test.modules_helpers import TwistedModulesMixin
from twisted.trial.unittest import SkipTest, TestCase
def test_installReactorMultiplePlugins(self):
    """
        Test that the L{reactors.installReactor} function correctly installs
        the specified reactor when there are multiple reactor plugins.
        """
    installed = []

    def install():
        installed.append(True)
    name = 'fakereactortest'
    package = __name__
    description = 'description'
    fakeReactor = FakeReactor(install, name, package, description)
    otherReactor = FakeReactor(lambda: None, 'otherreactor', package, description)
    self.pluginResults = [otherReactor, fakeReactor]
    reactors.installReactor(name)
    self.assertEqual(installed, [True])