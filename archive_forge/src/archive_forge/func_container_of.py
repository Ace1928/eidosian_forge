import sys
from typing import (
from mypy_extensions import mypyc_attr
from black.cache import CACHE_DIR
from black.mode import Mode, Preview
from black.strings import get_string_prefix, has_triple_quotes
from blib2to3 import pygram
from blib2to3.pgen2 import token
from blib2to3.pytree import NL, Leaf, Node, type_repr
def container_of(leaf: Leaf) -> LN:
    """Return `leaf` or one of its ancestors that is the topmost container of it.

    By "container" we mean a node where `leaf` is the very first child.
    """
    same_prefix = leaf.prefix
    container: LN = leaf
    while container:
        parent = container.parent
        if parent is None:
            break
        if parent.children[0].prefix != same_prefix:
            break
        if parent.type == syms.file_input:
            break
        if parent.prev_sibling is not None and parent.prev_sibling.type in BRACKETS:
            break
        container = parent
    return container