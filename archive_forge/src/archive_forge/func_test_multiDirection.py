import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_multiDirection(self):
    """
        Parsing a multi-direction mode setting with no parameters.
        """
    added, removed = irc.parseModes('+s-n+ti', [])
    self.assertEqual(added, [('s', None), ('t', None), ('i', None)])
    self.assertEqual(removed, [('n', None)])