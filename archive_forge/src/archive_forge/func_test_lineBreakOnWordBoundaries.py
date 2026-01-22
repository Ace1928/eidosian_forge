import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_lineBreakOnWordBoundaries(self):
    """
        IRCClient prefers to break long lines at word boundaries.
        """
    longline = 'o' * (irc.MAX_COMMAND_LENGTH // 2)
    self.client.msg('foo', longline + ' ' + longline)
    self.assertEqual(self.client.lines, ['PRIVMSG foo :' + longline, 'PRIVMSG foo :' + longline])