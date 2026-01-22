import array
from twisted.conch.insults import helper, insults
from twisted.python import text as tptext
def focusLost(self):
    return self.containee.focusLost()