import os
from io import StringIO
from typing import Sequence, Type
from unittest import skipIf
from zope.interface import Interface
from twisted import plugin
from twisted.cred import checkers, credentials, error, strcred
from twisted.plugins import cred_anonymous, cred_file, cred_unix
from twisted.python import usage
from twisted.python.fakepwd import UserDatabase
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
def test_findCheckerFactories(self):
    """
        L{strcred.findCheckerFactories} returns all available plugins.
        """
    availablePlugins = list(strcred.findCheckerFactories())
    for plg in plugin.getPlugins(strcred.ICheckerFactory):
        self.assertIn(plg, availablePlugins)