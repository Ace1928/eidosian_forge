from yowsup.common import YowConstants
from yowsup.layers.protocol_iq.protocolentities import IqProtocolEntity
from yowsup.structs import ProtocolTreeNode
def setGetStatusesProps(self, jids):
    assert type(jids) is list, 'jids must be a list of jids'
    self.jids = jids