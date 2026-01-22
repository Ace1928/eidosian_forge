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
def test_unixCheckerSucceedsBytes(self):
    """
        The checker works with valid L{bytes} credentials.
        """

    def _gotAvatar(username):
        self.assertEqual(username, self.adminBytes.username.decode('utf-8'))
    return self.checkerBytes.requestAvatarId(self.adminBytes).addCallback(_gotAvatar)