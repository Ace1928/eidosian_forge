import array
from twisted.conch.insults import helper, insults
from twisted.python import text as tptext
def func_PGUP(self, modifier):
    if self.renderOffset != 0:
        self.focusedIndex -= self.renderOffset
        self.renderOffset = 0
    else:
        self.focusedIndex = max(0, self.focusedIndex - self.height)
    self.repaint()