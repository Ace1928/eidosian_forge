import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_emptyModes(self):
    """
        Parsing an empty mode string raises L{irc.IRCBadModes}.
        """
    self.assertRaises(irc.IRCBadModes, irc.parseModes, '', [])