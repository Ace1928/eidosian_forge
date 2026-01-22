import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_resumeWhenFileDoesNotExist(self):
    """
        If given a resumeOffset to resume writing to a file that does not
        exist, L{DccFileReceive} will raise L{OSError}.
        """
    fp = FilePath(self.mktemp())
    error = self.assertRaises(OSError, self.makeConnectedDccFileReceive, fp.path, resumeOffset=1)
    self.assertEqual(errno.ENOENT, error.errno)