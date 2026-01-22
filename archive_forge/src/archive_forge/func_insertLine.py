from zope.interface import Interface, implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.python.compat import iterbytes, networkString
def insertLine(self, n=1):
    self.write(b'\x1b[%dL' % (n,))