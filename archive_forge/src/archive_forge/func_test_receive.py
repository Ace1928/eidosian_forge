from yowsup.layers import YowProtocolLayerTest
from yowsup.layers.protocol_chatstate import YowChatstateProtocolLayer
from yowsup.layers.protocol_chatstate.protocolentities import IncomingChatstateProtocolEntity, OutgoingChatstateProtocolEntity
def test_receive(self):
    entity = IncomingChatstateProtocolEntity(IncomingChatstateProtocolEntity.STATE_TYPING, 'jid@s.whatsapp.net')
    self.assertReceived(entity)