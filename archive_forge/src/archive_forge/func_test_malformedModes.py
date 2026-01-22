import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_malformedModes(self):
    """
        Parsing a mode string that does not start with C{+} or C{-} raises
        L{irc.IRCBadModes}.
        """
    self.assertRaises(irc.IRCBadModes, irc.parseModes, 'foo', [])
    self.assertRaises(irc.IRCBadModes, irc.parseModes, '%', [])