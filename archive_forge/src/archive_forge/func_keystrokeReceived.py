import array
from twisted.conch.insults import helper, insults
from twisted.python import text as tptext
def keystrokeReceived(self, keyID, modifier):
    return self.containee.keystrokeReceived(keyID, modifier)