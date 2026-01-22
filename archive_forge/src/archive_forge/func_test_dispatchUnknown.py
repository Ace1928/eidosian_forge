import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_dispatchUnknown(self):
    """
        Dispatching an unknown command invokes the default handler.
        """
    disp = Dispatcher()
    name = 'missing'
    args = (1, 2)
    res = disp.dispatch(name, *args)
    self.assertEqual(res, (name,) + args)