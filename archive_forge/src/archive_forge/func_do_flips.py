from .. import FatGraph, FatEdge, Link, Crossing
from ..links.links import CrossingEntryPoint
from ..links.ordered_set import OrderedSet
from .Base64LikeDT import (decode_base64_like_DT_code, encode_base64_like_DT_code)
def do_flips(self, v, v_edge, w, w_edge):
    """
        Decide whether v and/or w needs to be flipped in order to add
        an arc from v to w starting with the v_edge and ending with
        the w_edge.  If flips are needed, make them.  If the embedding
        cannot be extended raise an exception.
        """
    G = self.fat_graph
    vslot = G(v).index(v_edge)
    wslot = G(w).index(w_edge)
    for k in range(1, 3):
        ccw_edge = G(v)[vslot + k]
        if ccw_edge.marked:
            break
    if not ccw_edge.marked:
        raise ValueError('Invalid marking')
    left_slots = G.left_slots(ccw_edge)
    right_slots = G.right_slots(ccw_edge)
    v_valence, w_valence = (G.marked_valences[v], G.marked_valences[w])
    if (v, vslot) in left_slots:
        v_slot_side, v_other_side = (left_slots, right_slots)
    else:
        v_slot_side, v_other_side = (right_slots, left_slots)
    w_on_slot_side = w in [x[0] for x in v_slot_side]
    w_on_other_side = w in [x[0] for x in v_other_side]
    if not w_on_slot_side and (not w_on_other_side):
        raise EmbeddingError('Embedding does not extend.')
    if (w, wslot) in v_slot_side:
        if v_valence == w_valence == 2:
            G.push([v, w])
        return
    if w_valence != 2:
        G.flip(v)
        return
    elif v_valence != 2:
        G.flip(w)
        return
    if w_on_slot_side and (not w_on_other_side):
        G.flip(w)
        return
    if w_on_slot_side and w_on_other_side:
        G.push([w])
    G.flip(v)
    if not (w, wslot) in v_other_side:
        G.flip(w)