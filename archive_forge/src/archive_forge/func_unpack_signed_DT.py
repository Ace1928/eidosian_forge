from .. import FatGraph, FatEdge, Link, Crossing
from ..links.links import CrossingEntryPoint
from ..links.ordered_set import OrderedSet
from .Base64LikeDT import (decode_base64_like_DT_code, encode_base64_like_DT_code)
def unpack_signed_DT(self, signed_dt):
    dt = []
    component = []
    flips = []
    for byte in bytearray(signed_dt):
        flips.append(bool(byte & 1 << 6))
        label = (1 + byte & 31) << 1
        if byte & 1 << 5:
            label = -label
        component.append(label)
        if byte & 1 << 7:
            dt.append(tuple(component))
            component = []
    return (dt, flips)