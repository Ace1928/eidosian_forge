from yowsup.structs import ProtocolEntity, ProtocolTreeNode
def toProtocolTreeNode(self):
    return self._createProtocolTreeNode({'reason': self.reason})