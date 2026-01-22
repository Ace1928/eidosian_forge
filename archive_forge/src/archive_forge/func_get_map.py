from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def get_map(self, a_dict, maximum_size=100, search_key_func=None):
    c_map = self._get_map(a_dict, maximum_size=maximum_size, chk_bytes=self.get_chk_bytes(), search_key_func=search_key_func)
    return c_map