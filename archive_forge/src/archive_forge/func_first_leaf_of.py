import sys
from typing import (
from mypy_extensions import mypyc_attr
from black.cache import CACHE_DIR
from black.mode import Mode, Preview
from black.strings import get_string_prefix, has_triple_quotes
from blib2to3 import pygram
from blib2to3.pgen2 import token
from blib2to3.pytree import NL, Leaf, Node, type_repr
def first_leaf_of(node: LN) -> Optional[Leaf]:
    """Returns the first leaf of the node tree."""
    if isinstance(node, Leaf):
        return node
    if node.children:
        return first_leaf_of(node.children[0])
    else:
        return None