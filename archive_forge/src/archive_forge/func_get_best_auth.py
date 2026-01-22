import os
import struct
from Xlib import X, error
def get_best_auth(self, family, address, dispno, types=(b'MIT-MAGIC-COOKIE-1',)):
    """Find an authentication entry matching FAMILY, ADDRESS and
        DISPNO.

        The name of the auth scheme must match one of the names in
        TYPES.  If several entries match, the first scheme in TYPES
        will be choosen.

        If an entry is found, the tuple (name, data) is returned,
        otherwise XNoAuthError is raised.
        """
    num = str(dispno).encode()
    address = address.encode()
    matches = {}
    for efam, eaddr, enum, ename, edata in self.entries:
        if efam == family and eaddr == address and (num == enum):
            matches[ename] = edata
    for t in types:
        try:
            return (t, matches[t])
        except KeyError:
            pass
    raise error.XNoAuthError((family, address, dispno))