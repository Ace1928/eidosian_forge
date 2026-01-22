from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def begin_tagstart(self, byte):
    self.tagName = ''
    self.tagAttributes = {}
    self.termtag = 0
    self.endtag = 0