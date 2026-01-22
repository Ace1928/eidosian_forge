from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def begin_messyattr(self, byte):
    self.attrval = byte