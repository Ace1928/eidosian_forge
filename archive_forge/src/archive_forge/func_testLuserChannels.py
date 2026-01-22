import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def testLuserChannels(self):
    args = '7116'
    msg = 'channels formed'
    self._serverTestImpl('254', msg, 'luserChannels', args=args, channels=int(args))