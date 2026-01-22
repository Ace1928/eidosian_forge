from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def do_doctype(self, byte):
    if byte == '>':
        return 'bodydata'
    self.doctype += byte