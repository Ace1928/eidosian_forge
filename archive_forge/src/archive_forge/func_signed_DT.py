from .. import FatGraph, FatEdge, Link, Crossing
from ..links.links import CrossingEntryPoint
from ..links.ordered_set import OrderedSet
from .Base64LikeDT import (decode_base64_like_DT_code, encode_base64_like_DT_code)
def signed_DT(self):
    """
        Return a byte sequence containing the signed DT code.

        >>> d = DTcodec([(-6,-8,-2,-4)])
        >>> d2 = DTcodec(d.signed_DT())
        >>> d2.code
        [(-6, -8, -2, -4)]
        """
    code_bytes = bytearray()
    it = iter(self.flips)
    for component in self.code:
        for label in component:
            byte = abs(label)
            byte = (byte >> 1) - 1
            if label < 0:
                byte |= 1 << 5
            if next(it):
                byte |= 1 << 6
            code_bytes.append(byte)
        code_bytes[-1] |= 1 << 7
    return bytes(code_bytes)