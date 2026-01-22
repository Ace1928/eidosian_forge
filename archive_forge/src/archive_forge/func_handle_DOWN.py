import string
from typing import Dict
from zope.interface import implementer
from twisted.conch.insults import helper, insults
from twisted.logger import Logger
from twisted.python import reflect
from twisted.python.compat import iterbytes
def handle_DOWN(self):
    if self.historyPosition < len(self.historyLines) - 1:
        self.handle_HOME()
        self.terminal.eraseToLineEnd()
        self.historyPosition += 1
        self.lineBuffer = []
        self._deliverBuffer(self.historyLines[self.historyPosition])
    else:
        self.handle_HOME()
        self.terminal.eraseToLineEnd()
        self.historyPosition = len(self.historyLines)
        self.lineBuffer = []
        self.lineBufferIndex = 0