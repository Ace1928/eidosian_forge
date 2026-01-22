import array
from twisted.conch.insults import helper, insults
from twisted.python import text as tptext
def moveChild(self, child, x, y):
    for n in range(len(self.children)):
        if self.children[n][0] is child:
            self.children[n] = (child, x, y)
            break
    else:
        raise ValueError('No such child', child)