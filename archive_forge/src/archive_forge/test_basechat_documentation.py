from twisted.trial import unittest
from twisted.words.im import basechat, basesupport

        If an L{twisted.words.im.interfaces.IPerson} is in a
        L{basechat.Conversation}, L{basechat.ChatUI.contactChangedNick} causes a
        name change for that person in both the L{basechat.Conversation} and the
        L{basechat.ChatUI}.
        