from paste.util import intset
import socket
def itermasks(self):
    """Returns an iterator which iterates over ip/mask pairs which build
        this iprange if combined. An IP/Mask pair is returned in string form
        (e.g. '1.2.3.0/24')."""
    for r in self._ranges:
        for v in self._itermasks(r):
            yield v