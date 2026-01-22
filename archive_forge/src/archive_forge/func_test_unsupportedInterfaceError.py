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
def test_unsupportedInterfaceError(self):
    """
        The C{--auth} command line raises an exception when it
        gets a checker we don't support.
        """
    options = OptionsSupportsNoInterfaces()
    authType = cred_anonymous.theAnonymousCheckerFactory.authType
    self.assertRaises(usage.UsageError, options.parseOptions, ['--auth', authType])