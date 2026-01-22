from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def get_difference(self, new_roots, old_roots, search_key_func=None):
    if search_key_func is None:
        search_key_func = chk_map._search_key_plain
    return chk_map.CHKMapDifference(self.get_chk_bytes(), new_roots, old_roots, search_key_func)