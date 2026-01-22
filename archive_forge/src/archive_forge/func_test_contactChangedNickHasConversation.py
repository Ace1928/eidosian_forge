from twisted.trial import unittest
from twisted.words.im import basechat, basesupport
def test_contactChangedNickHasConversation(self) -> None:
    """
        If an L{twisted.words.im.interfaces.IPerson} is in a
        L{basechat.Conversation}, L{basechat.ChatUI.contactChangedNick} causes a
        name change for that person in both the L{basechat.Conversation} and the
        L{basechat.ChatUI}.
        """
    self.ui.persons[self.person.name, self.person.account] = self.person
    conversation = basechat.Conversation(self.person, self.ui)
    self.ui.conversations[self.person] = conversation
    self.assertEqual(self.person.name, 'foo')
    self.assertEqual(self.person.account, self.account)
    self.ui.contactChangedNick(self.person, 'bar')
    self.assertEqual(self.person.name, 'bar')
    self.assertEqual(self.person.account, self.account)