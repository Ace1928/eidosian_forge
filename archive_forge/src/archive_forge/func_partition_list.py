from .. import FatGraph, FatEdge, Link, Crossing
from ..links.links import CrossingEntryPoint
from ..links.ordered_set import OrderedSet
from .Base64LikeDT import (decode_base64_like_DT_code, encode_base64_like_DT_code)
def partition_list(L, parts):
    assert sum(parts) == len(L)
    ans = []
    k = 0
    for p in parts:
        ans.append(tuple(L[k:k + p]))
        k += p
    return ans