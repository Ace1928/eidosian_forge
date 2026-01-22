from twisted.internet.testing import StringTransport
from twisted.words.im.basechat import ChatUI, Conversation, GroupConversation
from twisted.words.im.ircsupport import IRCAccount, IRCProto
from twisted.words.im.locals import OfflineError
from twisted.words.test.test_irc import IRCTestCase
def test_sendGroupMessage(self) -> None:
    """
        L{IRCGroup.sendGroupMessage}
        """
    self.proto.makeConnection(self.transport)
    group = self.proto.chat.getGroup('#foo', self.proto)
    self.assertRaises(OfflineError, group.sendGroupMessage, 'Some message')
    group.account.client = self.proto
    self.transport.clear()
    group.sendGroupMessage('Some message 2')
    self.assertEqual(self.transport.io.getvalue(), b'PRIVMSG #foo :Some message 2\r\n')
    self.transport.clear()
    group.sendGroupMessage('smiles', {'style': 'emote'})
    self.assertEqual(self.transport.io.getvalue(), b'PRIVMSG #foo :\x01ACTION smiles\x01\r\n')