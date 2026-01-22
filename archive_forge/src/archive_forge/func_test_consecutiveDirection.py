import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_consecutiveDirection(self):
    """
        Parsing a multi-direction mode setting containing two consecutive mode
        sequences with the same direction results in the same result as if
        there were only one mode sequence in the same direction.
        """
    added, removed = irc.parseModes('+sn+ti', [])
    self.assertEqual(added, [('s', None), ('n', None), ('t', None), ('i', None)])
    self.assertEqual(removed, [])