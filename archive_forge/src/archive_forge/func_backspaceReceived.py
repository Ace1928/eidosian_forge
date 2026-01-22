import array
from twisted.conch.insults import helper, insults
from twisted.python import text as tptext
def backspaceReceived(self):
    if self.cursor > 0:
        self.buffer = self.buffer[:self.cursor - 1] + self.buffer[self.cursor:]
        self.cursor -= 1
        self.repaint()