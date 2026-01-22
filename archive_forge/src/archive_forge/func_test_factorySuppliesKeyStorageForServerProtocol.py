import struct
from twisted.test import iosim
from twisted.trial import unittest
from twisted.conch.error import ConchError, MissingKeyStoreError
from twisted.conch.test import keydata
def test_factorySuppliesKeyStorageForServerProtocol(self):
    msg = struct.pack('!LB', 1, agent.AGENTC_REQUEST_IDENTITIES)
    del self.server.factory.__dict__['keys']
    self.assertRaises(MissingKeyStoreError, self.server.dataReceived, msg)