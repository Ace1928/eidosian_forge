from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def begin_bodydata(self, byte):
    if self._leadingBodyData:
        self.bodydata = self._leadingBodyData
        del self._leadingBodyData
    else:
        self.bodydata = ''