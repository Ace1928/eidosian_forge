from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def end_cdata(self):
    self.gotCData(self.cdatabuf)
    self.cdatabuf = ''