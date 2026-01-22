import collections
import dns.name
from ._compat import xrange
def get_deepest_match(self, name):
    """Find the deepest match to *fname* in the dictionary.

        The deepest match is the longest name in the dictionary which is
        a superdomain of *name*.  Note that *superdomain* includes matching
        *name* itself.

        *name*, a ``dns.name.Name``, the name to find.

        Returns a ``(key, value)`` where *key* is the deepest
        ``dns.name.Name``, and *value* is the value associated with *key*.
        """
    depth = len(name)
    if depth > self.max_depth:
        depth = self.max_depth
    for i in xrange(-depth, 0):
        n = dns.name.Name(name[i:])
        if n in self:
            return (n, self[n])
    v = self[dns.name.empty]
    return (dns.name.empty, v)