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
def test_displaysListCorrectly(self):
    """
        The C{--help-auth} argument correctly displays all
        available authentication plugins, then exits.
        """
    newStdout = StringIO()
    options = DummyOptions()
    options.authOutput = newStdout
    self.assertRaises(SystemExit, options.parseOptions, ['--help-auth'])
    for checkerFactory in strcred.findCheckerFactories():
        self.assertIn(checkerFactory.authType, newStdout.getvalue())