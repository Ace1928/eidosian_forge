from twisted.trial import unittest
from twisted.words.protocols.jabber import jid
def test_userhostOnlyHost(self) -> None:
    """
        Test the extraction of the bare JID of the full form host/resource.
        """
    j = jid.JID('host/resource')
    self.assertEqual('host', j.userhost())