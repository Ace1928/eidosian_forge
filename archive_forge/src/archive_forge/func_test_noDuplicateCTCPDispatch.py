import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_noDuplicateCTCPDispatch(self):
    """
        Duplicated CTCP messages are ignored and no reply is made.
        """

    def testCTCP(user, channel, data):
        self.called += 1
    self.called = 0
    self.client.ctcpQuery_TESTTHIS = testCTCP
    self.client.irc_PRIVMSG('foo!bar@baz.quux', ['#chan', '{X}TESTTHIS{X}foo{X}TESTTHIS{X}'.format(X=irc.X_DELIM)])
    self.assertEqualBufferValue(self.file.getvalue(), '')
    self.assertEqual(self.called, 1)