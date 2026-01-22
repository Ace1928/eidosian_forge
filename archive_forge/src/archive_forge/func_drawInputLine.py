import string
from typing import Dict
from zope.interface import implementer
from twisted.conch.insults import helper, insults
from twisted.logger import Logger
from twisted.python import reflect
from twisted.python.compat import iterbytes
def drawInputLine(self):
    """
        Write a line containing the current input prompt and the current line
        buffer at the current cursor position.
        """
    self.terminal.write(self.ps[self.pn] + b''.join(self.lineBuffer))