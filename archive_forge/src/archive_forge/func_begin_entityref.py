from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def begin_entityref(self, byte):
    self.erefbuf = ''
    self.erefextra = ''