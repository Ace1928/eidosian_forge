from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def end_unentity(self):
    self.gotText(self.bodydata)