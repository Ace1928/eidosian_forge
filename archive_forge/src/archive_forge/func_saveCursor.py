from zope.interface import Interface, implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.python.compat import iterbytes, networkString
def saveCursor(self):
    self._savedCursorPos = Vector(self.cursorPos.x, self.cursorPos.y)
    self.write(b'\x1b7')