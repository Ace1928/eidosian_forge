import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_sufficientWidth_msg(self):
    """
        Messages exactly equal in length to the C{length} parameter to
        L{IRCClient.msg} are sent in a single command.
        """
    msg = 'barbazbo'
    maxLen = len(f'PRIVMSG foo :{msg}') + 2
    self.client.msg('foo', msg, maxLen)
    self.assertEqual(self.client.lines, [f'PRIVMSG foo :{msg}'])
    self.client.lines = []
    self.client.msg('foo', msg, maxLen - 1)
    self.assertEqual(2, len(self.client.lines))
    self.client.lines = []
    self.client.msg('foo', msg, maxLen + 1)
    self.assertEqual(1, len(self.client.lines))