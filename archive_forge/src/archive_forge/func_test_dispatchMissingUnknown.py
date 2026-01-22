import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_dispatchMissingUnknown(self):
    """
        Dispatching an unknown command, when no default handler is present,
        results in an exception being raised.
        """
    disp = Dispatcher()
    disp.disp_unknown = None
    self.assertRaises(irc.UnhandledCommand, disp.dispatch, 'bar')