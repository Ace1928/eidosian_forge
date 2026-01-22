from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
def make_one_deep_two_prefix_map(self, search_key_func=None):
    """Create a map with one internal node, but references are extra long.

        Otherwise has similar content to make_two_deep_map.
        """
    return self.get_map({(b'aaa',): b'initial aaa content', (b'add',): b'initial add content', (b'adh',): b'initial adh content', (b'adl',): b'initial adl content'}, search_key_func=search_key_func)