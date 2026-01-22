import array
from twisted.conch.insults import helper, insults
from twisted.python import text as tptext
def cursorHome(self):
    return self.terminal.cursorPosition(self.xoff, self.yoff)