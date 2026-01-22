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
def test_failsWithEmptyFilename(self):
    """
        An empty filename raises an error.
        """
    self.assertRaises(ValueError, strcred.makeChecker, 'file')
    self.assertRaises(ValueError, strcred.makeChecker, 'file:')