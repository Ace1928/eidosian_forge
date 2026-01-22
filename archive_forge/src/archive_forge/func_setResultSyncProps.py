from yowsup.structs import ProtocolTreeNode
from .iq_sync import SyncIqProtocolEntity
def setResultSyncProps(self, version, inNumbers, outNumbers, invalidNumbers, wait=None):
    assert type(inNumbers) is dict, 'in numbers must be a dict {number -> jid}'
    assert type(outNumbers) is dict, 'out numbers must be a dict {number -> jid}'
    assert type(invalidNumbers) is list, 'invalid numbers must be a list'
    self.inNumbers = inNumbers
    self.outNumbers = outNumbers
    self.invalidNumbers = invalidNumbers
    self.wait = int(wait) if wait is not None else None
    self.version = version