import struct
from twisted.test import iosim
from twisted.trial import unittest
from twisted.conch.error import ConchError, MissingKeyStoreError
from twisted.conch.test import keydata
def test_agentc_REMOVE_ALL_RSA_IDENTITIES(self):
    """
        assert that we get the correct op code for an RSA remove all identities
        request.
        """
    d = self.client.sendRequest(agent.AGENTC_REMOVE_ALL_RSA_IDENTITIES, b'')
    self.pump.flush()
    return d.addCallback(self.assertEqual, b'')