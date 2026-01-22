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
def test_canAddSupportedChecker(self):
    """
        When addChecker is called with a checker that implements at least one
        of the interfaces our application supports, it is successful.
        """
    options = OptionsForUsernamePassword()
    options.addChecker(self.goodChecker)
    iface = options.supportedInterfaces[0]
    self.assertIdentical(options['credInterfaces'][iface][0], self.goodChecker)
    self.assertIdentical(options['credCheckers'][0], self.goodChecker)
    self.assertEqual(len(options['credInterfaces'][iface]), 1)
    self.assertEqual(len(options['credCheckers']), 1)