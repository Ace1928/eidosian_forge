from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def make_one_deep_one_prefix_map(self, search_key_func=None):
    """Create a map with one internal node, but references are extra long.

        Similar to make_one_deep_two_prefix_map, except the split is at the
        first char, rather than the second.
        """
    return self.get_map({(b'add',): b'initial add content', (b'adh',): b'initial adh content', (b'adl',): b'initial adl content', (b'bbb',): b'initial bbb content'}, search_key_func=search_key_func)