from io import BytesIO
import struct
import sys
import copy
import encodings.idna
import dns.exception
import dns.wiredata
from ._compat import long, binary_type, text_type, unichr, maybe_decode
def is_wild(self):
    """Is this name wild?  (I.e. Is the least significant label '*'?)

        Returns a ``bool``.
        """
    return len(self.labels) > 0 and self.labels[0] == b'*'