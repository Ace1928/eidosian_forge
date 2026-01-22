import sys
from typing import (
from mypy_extensions import mypyc_attr
from black.cache import CACHE_DIR
from black.mode import Mode, Preview
from black.strings import get_string_prefix, has_triple_quotes
from blib2to3 import pygram
from blib2to3.pgen2 import token
from blib2to3.pytree import NL, Leaf, Node, type_repr
def parent_type(node: Optional[LN]) -> Optional[NodeType]:
    """
    Returns:
        @node.parent.type, if @node is not None and has a parent.
            OR
        None, otherwise.
    """
    if node is None or node.parent is None:
        return None
    return node.parent.type