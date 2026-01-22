import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_ERRMSG(self):
    """Testing CTCP query ERRMSG.

        Not because this is this is an especially important case in the
        field, but it does go through the entire dispatch/decode/encode
        process.
        """
    errQuery = ':nick!guy@over.there PRIVMSG #theChan :%(X)cERRMSG t%(X)c%(EOL)s' % {'X': irc.X_DELIM, 'EOL': irc.CR + irc.LF}
    errReply = 'NOTICE nick :%(X)cERRMSG t :No error has occurred.%(X)c%(EOL)s' % {'X': irc.X_DELIM, 'EOL': irc.CR + irc.LF}
    self.client.dataReceived(errQuery)
    reply = self.file.getvalue()
    self.assertEqualBufferValue(reply, errReply)