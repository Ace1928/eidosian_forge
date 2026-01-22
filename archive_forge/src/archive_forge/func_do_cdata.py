from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def do_cdata(self, byte):
    self.cdatabuf += byte
    if self.cdatabuf.endswith(']]>'):
        self.cdatabuf = self.cdatabuf[:-3]
        return 'bodydata'