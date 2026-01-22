from zope.interface import Interface, implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.python.compat import iterbytes, networkString
def selectGraphicRendition(self, *attributes):
    attrs = []
    for a in attributes:
        attrs.append(networkString(a))
    self.write(b'\x1b[' + b';'.join(attrs) + b'm')