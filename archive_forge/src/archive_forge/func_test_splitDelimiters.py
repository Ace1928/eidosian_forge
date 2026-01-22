import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_splitDelimiters(self):
    """
        L{twisted.words.protocols.irc.split} skips any delimiter (space or
        newline) that it finds at the very beginning of the string segment it
        is operating on.  Nothing should be added to the output list because of
        it.
        """
    r = irc.split('xx yyz', 2)
    self.assertEqual(['xx', 'yy', 'z'], r)
    r = irc.split('xx\nyyz', 2)
    self.assertEqual(['xx', 'yy', 'z'], r)