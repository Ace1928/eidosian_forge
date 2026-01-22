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
def test_findCheckerFactory(self):
    """
        L{strcred.findCheckerFactory} returns the first plugin
        available for a given authentication type.
        """
    self.assertIdentical(strcred.findCheckerFactory('file'), cred_file.theFileCheckerFactory)