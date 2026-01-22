from yowsup.common import YowConstants
from yowsup.layers.protocol_iq.protocolentities import ResultIqProtocolEntity
from yowsup.structs import ProtocolTreeNode
def setUploadProps(self, url, ip=None, resumeOffset=0, duplicate=False):
    self.url = url
    self.ip = ip
    self.resumeOffset = resumeOffset or 0
    self.duplicate = duplicate