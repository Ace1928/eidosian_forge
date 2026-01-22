from __future__ import annotations
from twisted.cred import credentials, error
from twisted.cred.checkers import FilePasswordDB
from twisted.internet import defer
from twisted.trial import unittest
from twisted.words import tap
def test_hostname(self) -> None:
    """
        Tests that the --hostname parameter gets passed to Options.
        """
    opt = tap.Options()
    opt.parseOptions(['--hostname', 'myhost'])
    self.assertEqual(opt['hostname'], 'myhost')