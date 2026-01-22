from twisted.trial import unittest
from twisted.words.protocols.jabber import jid
def test_fullHost(self) -> None:
    """
        Test giving a string representation of the JID with only a host part.
        """
    j = jid.JID(tuple=(None, 'host', None))
    self.assertEqual('host', j.full())