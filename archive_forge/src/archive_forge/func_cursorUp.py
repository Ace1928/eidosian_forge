from zope.interface import Interface, implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.python.compat import iterbytes, networkString
def cursorUp(self, n=1):
    assert n >= 1
    self.cursorPos.y = max(self.cursorPos.y - n, 0)
    self.write(b'\x1b[%dA' % (n,))