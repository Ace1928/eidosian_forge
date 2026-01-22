import string
from typing import Dict
from zope.interface import implementer
from twisted.conch.insults import helper, insults
from twisted.logger import Logger
from twisted.python import reflect
from twisted.python.compat import iterbytes
def handle_TAB(self):
    n = self.TABSTOP - len(self.lineBuffer) % self.TABSTOP
    self.terminal.cursorForward(n)
    self.lineBufferIndex += n
    self.lineBuffer.extend(iterbytes(b' ' * n))