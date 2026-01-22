from twisted.internet.testing import StringTransport
from twisted.words.im.basechat import ChatUI, Conversation, GroupConversation
from twisted.words.im.ircsupport import IRCAccount, IRCProto
from twisted.words.im.locals import OfflineError
from twisted.words.test.test_irc import IRCTestCase
def test_isupport(self) -> None:
    """
        L{IRCProto} can interpret I{ISUPPORT} (I{005}) messages from the server
        and reflect their information in its C{supported} attribute.
        """
    self.proto.makeConnection(self.transport)
    self.proto.dataReceived(':irc.example.com 005 alice MODES=4 CHANLIMIT=#:20\r\n')
    self.assertEqual(4, self.proto.supported.getFeature('MODES'))