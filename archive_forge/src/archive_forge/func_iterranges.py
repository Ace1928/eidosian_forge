from paste.util import intset
import socket
def iterranges(self):
    """Returns an iterator which iterates over ip-ip ranges which build
        this iprange if combined. An ip-ip pair is returned in string form
        (e.g. '1.2.3.4-2.3.4.5')."""
    for r in self._ranges:
        if r[1] - r[0] == 1:
            yield self._int2ip(r[0])
        else:
            yield ('%s-%s' % (self._int2ip(r[0]), self._int2ip(r[1] - 1)))