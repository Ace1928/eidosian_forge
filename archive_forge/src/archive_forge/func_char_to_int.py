from .. import FatGraph, FatEdge, Link, Crossing
from ..links.links import CrossingEntryPoint
from ..links.ordered_set import OrderedSet
from .Base64LikeDT import (decode_base64_like_DT_code, encode_base64_like_DT_code)
def char_to_int(x):
    n = ord(x)
    if 96 < n < 123:
        return n - 96
    if 64 < n < 91:
        return 64 - n
    raise ValueError('Not an ascii letter.')