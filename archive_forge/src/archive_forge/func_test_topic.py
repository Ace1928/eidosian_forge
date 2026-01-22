from twisted.internet.testing import StringTransport
from twisted.words.im.basechat import ChatUI, Conversation, GroupConversation
from twisted.words.im.ircsupport import IRCAccount, IRCProto
from twisted.words.im.locals import OfflineError
from twisted.words.test.test_irc import IRCTestCase
def test_topic(self) -> None:
    """
        IRC TOPIC command changes the topic of an IRC channel.
        """
    self.proto.makeConnection(self.transport)
    self.proto.dataReceived(':alice1 JOIN #group1\r\n')
    self.proto.dataReceived(':alice1 TOPIC #group1 newtopic\r\n')
    groupConversation = self.proto.getGroupConversation('group1')
    self.assertEqual(groupConversation.topic, 'newtopic')
    self.assertEqual(groupConversation.topicSetBy, 'alice1')