from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def do_attrval(self, byte):
    if byte == self.quotetype:
        return 'attrs'
    self.attrval += byte