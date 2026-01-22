import struct
from twisted.test import iosim
from twisted.trial import unittest
from twisted.conch.error import ConchError, MissingKeyStoreError
from twisted.conch.test import keydata
def test_agentc_REQUEST_RSA_IDENTITIES(self):
    """
        assert that we get the correct op code for an RSA identities request
        """
    d = self.client.sendRequest(agent.AGENTC_REQUEST_RSA_IDENTITIES, b'')
    self.pump.flush()

    def _cb(packet):
        self.assertEqual(agent.AGENT_RSA_IDENTITIES_ANSWER, ord(packet[0:1]))
    return d.addCallback(_cb)