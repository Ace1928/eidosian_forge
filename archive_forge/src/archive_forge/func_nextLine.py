from zope.interface import Interface, implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.python.compat import iterbytes, networkString
def nextLine(self):
    self.cursorPos.x = 0
    self.cursorPos.y = min(self.cursorPos.y + 1, self.termSize.y - 1)
    self.write(b'\n')