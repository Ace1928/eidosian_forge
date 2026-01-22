import collections
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
@property
def support_set(self):
    """Returns the set of simple symbols that this QN relies on.

    This would be the smallest set of symbols necessary for the QN to
    statically resolve (assuming properties and index ranges are verified
    at runtime).

    Examples:
      'a.b' has only one support symbol, 'a'
      'a[i]' has two support symbols, 'a' and 'i'
    """
    roots = set()
    if self.has_attr():
        roots.update(self.parent.support_set)
    elif self.has_subscript():
        roots.update(self.parent.support_set)
        roots.update(self.qn[1].support_set)
    else:
        roots.add(self)
    return roots