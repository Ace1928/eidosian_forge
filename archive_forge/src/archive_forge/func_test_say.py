import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_say(self):
    """
        L{IRCClient.say} prepends the channel prefix C{"#"} if necessary and
        then sends the message to the server for delivery to that channel.
        """
    self.client.say('thechannel', 'the message')
    self.assertEqual(self.client.lines, ['PRIVMSG #thechannel :the message'])