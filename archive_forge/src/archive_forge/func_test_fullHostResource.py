from twisted.trial import unittest
from twisted.words.protocols.jabber import jid
def test_fullHostResource(self) -> None:
    """
        Test giving a string representation of the JID with host, resource.
        """
    j = jid.JID(tuple=(None, 'host', 'resource'))
    self.assertEqual('host/resource', j.full())