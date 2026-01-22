from twisted.internet.testing import StringTransport
from twisted.words.im.basechat import ChatUI, Conversation, GroupConversation
from twisted.words.im.ircsupport import IRCAccount, IRCProto
from twisted.words.im.locals import OfflineError
from twisted.words.test.test_irc import IRCTestCase
def test_channels(self) -> None:
    """
        If created with an account with a list of channels, L{IRCProto}
        joins each of those channels after registering.
        """
    self.account.channels = ['#foo', '#bar']
    self.proto.makeConnection(self.transport)
    self.assertEqualBufferValue(self.transport.value(), 'NICK alice\r\nUSER alice foo bar :Twisted-IM user\r\nJOIN #foo\r\nJOIN #bar\r\n')