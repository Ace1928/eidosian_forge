from yowsup.layers.protocol_contacts.protocolentities.iq_sync_result import ResultSyncIqProtocolEntity
from yowsup.structs.protocolentity import ProtocolEntityTest
import unittest
def test_delta_result(self):
    del self.node.getChild('sync')['wait']
    self.test_generation()