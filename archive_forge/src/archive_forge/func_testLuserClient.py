import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def testLuserClient(self):
    msg = 'There are 9227 victims and 9542 hiding on 24 servers'
    self._serverTestImpl('251', msg, 'luserClient', info=msg)