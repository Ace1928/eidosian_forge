from yowsup.structs import ProtocolTreeNode
from yowsup.layers.protocol_iq.protocolentities import IqProtocolEntity
import time
def setSyncProps(self, sid, index, last):
    self.sid = sid if sid else str((int(time.time()) + 11644477200) * 10000000)
    self.index = int(index)
    self.last = last