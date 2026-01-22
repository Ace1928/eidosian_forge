import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_resumeFromResumeOffset(self):
    """
        If given a resumeOffset argument, L{DccFileReceive} will attempt to
        resume from that number of bytes if the file exists.
        """
    fp = FilePath(self.mktemp())
    fp.setContent(b'Twisted is awesome!')
    protocol = self.makeConnectedDccFileReceive(fp.path, resumeOffset=11)
    self.allDataReceivedForProtocol(protocol, b'amazing!')
    self.assertEqual(fp.getContent(), b'Twisted is amazing!')