from twisted.internet.testing import StringTransport
from twisted.words.im.basechat import ChatUI, Conversation, GroupConversation
from twisted.words.im.ircsupport import IRCAccount, IRCProto
from twisted.words.im.locals import OfflineError
from twisted.words.test.test_irc import IRCTestCase
def test_quit(self) -> None:
    """
        IRC QUIT command removes a user from all IRC channels.
        """
    self.proto.makeConnection(self.transport)
    self.proto.dataReceived(':alice1 JOIN #group1\r\n')
    self.assertIn('group1', self.proto._ingroups['alice1'])
    self.assertNotIn('group2', self.proto._ingroups['alice1'])
    self.proto.dataReceived(':alice1 JOIN #group3\r\n')
    self.assertIn('group3', self.proto._ingroups['alice1'])
    self.proto.dataReceived(':alice1 QUIT\r\n')
    self.assertTrue(len(self.proto._ingroups['alice1']) == 0)
    self.proto.dataReceived(':alice3 QUIT\r\n')