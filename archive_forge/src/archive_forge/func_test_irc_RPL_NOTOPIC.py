import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_irc_RPL_NOTOPIC(self):
    """
        L{IRCClient.topicUpdated} is called when the topic is removed.
        """
    self.client.irc_RPL_NOTOPIC(self.user, ['?', self.channel])
    self.assertEqual(self.client.methods, [('topicUpdated', ('Wolf', self.channel, ''))])