from twisted.trial import unittest
from twisted.words.protocols.jabber import jid
def test_userhostJIDNoResource(self) -> None:
    """
        Test getting a JID object of the bare JID when there was no resource.
        """
    j = jid.JID('user@host')
    self.assertIdentical(j, j.userhostJID())