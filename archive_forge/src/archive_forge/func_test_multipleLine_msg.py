import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_multipleLine_msg(self):
    """
        Messages longer than the C{length} parameter to L{IRCClient.msg} will
        be split and sent in multiple commands.
        """
    maxLen_msg = len('PRIVMSG foo :') + 3 + 2
    self.client.msg('foo', 'barbazbo', maxLen_msg)
    self.assertEqual(self.client.lines, ['PRIVMSG foo :bar', 'PRIVMSG foo :baz', 'PRIVMSG foo :bo'])