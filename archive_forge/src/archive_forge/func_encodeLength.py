import sys
from pyasn1 import debug
from pyasn1 import error
from pyasn1.codec.ber import eoo
from pyasn1.compat import _MISSING
from pyasn1.compat.integer import to_bytes
from pyasn1.compat.octets import (int2oct, oct2int, ints2octs, null,
from pyasn1.type import char
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful
def encodeLength(self, length, defMode):
    if not defMode and self.supportIndefLenMode:
        return (128,)
    if length < 128:
        return (length,)
    else:
        substrate = ()
        while length:
            substrate = (length & 255,) + substrate
            length >>= 8
        substrateLen = len(substrate)
        if substrateLen > 126:
            raise error.PyAsn1Error('Length octets overflow (%d)' % substrateLen)
        return (128 | substrateLen,) + substrate