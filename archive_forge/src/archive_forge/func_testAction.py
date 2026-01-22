import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def testAction(self):
    self.p.action('this-is-sender', 'this-is-recip', 'this is action')
    self.check(':this-is-sender ACTION this-is-recip :this is action\r\n')