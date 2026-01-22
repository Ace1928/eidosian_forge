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
def test_reactorSelectionMixin(self):
    """
        Test that the reactor selected is installed as soon as possible, ie
        when the option is parsed.
        """
    executed = []
    INSTALL_EVENT = 'reactor installed'
    SUBCOMMAND_EVENT = 'subcommands loaded'

    class ReactorSelectionOptions(usage.Options, app.ReactorSelectionMixin):

        @property
        def subCommands(self):
            executed.append(SUBCOMMAND_EVENT)
            return [('subcommand', None, lambda: self, 'test subcommand')]

    def install():
        executed.append(INSTALL_EVENT)
    self.pluginResults = [FakeReactor(install, 'fakereactortest', __name__, 'described')]
    options = ReactorSelectionOptions()
    options.parseOptions(['--reactor', 'fakereactortest', 'subcommand'])
    self.assertEqual(executed[0], INSTALL_EVENT)
    self.assertEqual(executed.count(INSTALL_EVENT), 1)
    self.assertEqual(options['reactor'], 'fakereactortest')