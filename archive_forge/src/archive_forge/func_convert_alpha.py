from .. import FatGraph, FatEdge, Link, Crossing
from ..links.links import CrossingEntryPoint
from ..links.ordered_set import OrderedSet
from .Base64LikeDT import (decode_base64_like_DT_code, encode_base64_like_DT_code)
def convert_alpha(self, code):
    code = string_to_ints(code)
    num_crossings, components = code[:2]
    comp_lengths = code[2:2 + components]
    crossings = [x << 1 for x in code[2 + components:]]
    assert len(crossings) == num_crossings
    return partition_list(crossings, comp_lengths)