from twisted.internet.testing import StringTransport
from twisted.words.im.basechat import ChatUI, Conversation, GroupConversation
from twisted.words.im.ircsupport import IRCAccount, IRCProto
from twisted.words.im.locals import OfflineError
from twisted.words.test.test_irc import IRCTestCase
def test_rplTopic(self) -> None:
    """
        RPL_TOPIC server response (332) is sent when a channel's topic is changed
        """
    self.proto.makeConnection(self.transport)
    self.proto.dataReceived(':example.com 332 alice, #foo :Some random topic\r\n')
    self.assertEqual('Some random topic', self.proto._topics['foo'])