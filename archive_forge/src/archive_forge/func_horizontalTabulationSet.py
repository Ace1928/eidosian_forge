from zope.interface import Interface, implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.python.compat import iterbytes, networkString
def horizontalTabulationSet(self):
    self.write(b'\x1bH')