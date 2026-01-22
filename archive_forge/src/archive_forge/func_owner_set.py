import collections
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
@property
def owner_set(self):
    """Returns all the symbols (simple or composite) that own this QN.

    In other words, if this symbol was modified, the symbols in the owner set
    may also be affected.

    Examples:
      'a.b[c.d]' has two owners, 'a' and 'a.b'
    """
    owners = set()
    if self.has_attr() or self.has_subscript():
        owners.add(self.parent)
        owners.update(self.parent.owner_set)
    return owners