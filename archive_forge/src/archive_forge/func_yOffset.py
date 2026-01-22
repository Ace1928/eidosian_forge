import array
from twisted.conch.insults import helper, insults
from twisted.python import text as tptext
@yOffset.setter
def yOffset(self, value):
    if self._yOffset != value:
        self._yOffset = value
        self.repaint()