import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_singleMode(self):
    """
        Parsing a single mode setting with no parameters results in that mode,
        with no parameters, in the "added" direction and no modes in the
        "removed" direction.
        """
    added, removed = irc.parseModes('+s', [])
    self.assertEqual(added, [('s', None)])
    self.assertEqual(removed, [])
    added, removed = irc.parseModes('-s', [])
    self.assertEqual(added, [])
    self.assertEqual(removed, [('s', None)])