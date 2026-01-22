from twisted.internet.testing import StringTransport
from twisted.words.im.basechat import ChatUI, Conversation, GroupConversation
from twisted.words.im.ircsupport import IRCAccount, IRCProto
from twisted.words.im.locals import OfflineError
from twisted.words.test.test_irc import IRCTestCase
def test_sendMessage(self) -> None:
    """
        L{IRCPerson.sendMessage}
        """
    self.proto.makeConnection(self.transport)
    person = self.proto.getPerson('alice')
    self.assertRaises(OfflineError, person.sendMessage, 'Some message')
    person.account.client = self.proto
    self.transport.clear()
    person.sendMessage('Some message 2')
    self.assertEqual(self.transport.io.getvalue(), b'PRIVMSG alice :Some message 2\r\n')
    self.transport.clear()
    person.sendMessage('smiles', {'style': 'emote'})
    self.assertEqual(self.transport.io.getvalue(), b'PRIVMSG alice :\x01ACTION smiles\x01\r\n')