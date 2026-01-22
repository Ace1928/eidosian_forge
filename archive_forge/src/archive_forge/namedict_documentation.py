import collections
import dns.name
from ._compat import xrange
Find the deepest match to *fname* in the dictionary.

        The deepest match is the longest name in the dictionary which is
        a superdomain of *name*.  Note that *superdomain* includes matching
        *name* itself.

        *name*, a ``dns.name.Name``, the name to find.

        Returns a ``(key, value)`` where *key* is the deepest
        ``dns.name.Name``, and *value* is the value associated with *key*.
        