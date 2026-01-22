from .. import FatGraph, FatEdge, Link, Crossing
from ..links.links import CrossingEntryPoint
from ..links.ordered_set import OrderedSet
from .Base64LikeDT import (decode_base64_like_DT_code, encode_base64_like_DT_code)
def marked_valence(self, vertex):
    """
        Compute the marked valence of a vertex.
        """
    valence = 0
    for e in self.incidence_dict[vertex]:
        if e.marked:
            valence += 1
    return valence