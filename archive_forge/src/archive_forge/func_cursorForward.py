from zope.interface import Interface, implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.python.compat import iterbytes, networkString
def cursorForward(self, n=1):
    assert n >= 1
    self.cursorPos.x = min(self.cursorPos.x + n, self.termSize.x - 1)
    self.write(b'\x1b[%dC' % (n,))