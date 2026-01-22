from twisted.trial import unittest
from twisted.words.protocols.jabber import jid
def test_userhostJID(self) -> None:
    """
        Test getting a JID object of the bare JID.
        """
    j1 = jid.JID('user@host/resource')
    j2 = jid.internJID('user@host')
    self.assertIdentical(j2, j1.userhostJID())