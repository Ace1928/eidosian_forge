from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def begin_afterslash(self, byte):
    self._after_slash_closed = 0