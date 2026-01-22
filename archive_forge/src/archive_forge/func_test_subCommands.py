import errno
import inspect
import os
import pickle
import signal
import sys
from io import StringIO
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted import internet, logger, plugin
from twisted.application import app, reactors, service
from twisted.application.service import IServiceMaker
from twisted.internet.base import ReactorBase
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IReactorDaemonize, _ISupportsExitSignalCapturing
from twisted.internet.test.modulehelpers import AlternateReactor
from twisted.internet.testing import MemoryReactor
from twisted.logger import ILogObserver, globalLogBeginner, globalLogPublisher
from twisted.python import util
from twisted.python.components import Componentized
from twisted.python.fakepwd import UserDatabase
from twisted.python.log import ILogObserver as LegacyILogObserver, textFromEventDict
from twisted.python.reflect import requireModule
from twisted.python.runtime import platformType
from twisted.python.usage import UsageError
from twisted.scripts import twistd
from twisted.test.test_process import MockOS
from twisted.trial.unittest import TestCase
def test_subCommands(self):
    """
        subCommands is built from IServiceMaker plugins, and is sorted
        alphabetically.
        """

    class FakePlugin:

        def __init__(self, name):
            self.tapname = name
            self._options = 'options for ' + name
            self.description = 'description of ' + name

        def options(self):
            return self._options
    apple = FakePlugin('apple')
    banana = FakePlugin('banana')
    coconut = FakePlugin('coconut')
    donut = FakePlugin('donut')

    def getPlugins(interface):
        self.assertEqual(interface, IServiceMaker)
        yield coconut
        yield banana
        yield donut
        yield apple
    config = twistd.ServerOptions()
    self.assertEqual(config._getPlugins, plugin.getPlugins)
    config._getPlugins = getPlugins
    subCommands = config.subCommands
    expectedOrder = [apple, banana, coconut, donut]
    for subCommand, expectedCommand in zip(subCommands, expectedOrder):
        name, shortcut, parserClass, documentation = subCommand
        self.assertEqual(name, expectedCommand.tapname)
        self.assertIsNone(shortcut)
        (self.assertEqual(parserClass(), expectedCommand._options),)
        self.assertEqual(documentation, expectedCommand.description)