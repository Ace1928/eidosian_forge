from .. import FatGraph, FatEdge, Link, Crossing
from ..links.links import CrossingEntryPoint
from ..links.ordered_set import OrderedSet
from .Base64LikeDT import (decode_base64_like_DT_code, encode_base64_like_DT_code)
def right_slots(self, edge):
    """
        Return the (vertex, slot) pairs on the right boundary curve.
        """
    return set(self._boundary_slots(edge, side=1))