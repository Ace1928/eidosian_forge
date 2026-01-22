import sys
from typing import (
from mypy_extensions import mypyc_attr
from black.cache import CACHE_DIR
from black.mode import Mode, Preview
from black.strings import get_string_prefix, has_triple_quotes
from blib2to3 import pygram
from blib2to3.pgen2 import token
from blib2to3.pytree import NL, Leaf, Node, type_repr
def first_leaf(node: LN) -> Optional[Leaf]:
    """Returns the first leaf of the ancestor node."""
    if isinstance(node, Leaf):
        return node
    elif not node.children:
        return None
    else:
        return first_leaf(node.children[0])