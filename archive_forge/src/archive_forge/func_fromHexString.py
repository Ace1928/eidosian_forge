import math
import sys
from pyasn1 import error
from pyasn1.codec.ber import eoo
from pyasn1.compat import integer
from pyasn1.compat import octets
from pyasn1.type import base
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import tagmap
@staticmethod
def fromHexString(value):
    """Create a |ASN.1| object initialized from the hex string.

        Parameters
        ----------
        value: :class:`str`
            Text string like 'DEADBEEF'
        """
    r = []
    p = []
    for v in value:
        if p:
            r.append(int(p + v, 16))
            p = None
        else:
            p = v
    if p:
        r.append(int(p + '0', 16))
    return octets.ints2octs(r)