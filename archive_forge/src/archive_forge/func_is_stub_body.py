import sys
from typing import (
from mypy_extensions import mypyc_attr
from black.cache import CACHE_DIR
from black.mode import Mode, Preview
from black.strings import get_string_prefix, has_triple_quotes
from blib2to3 import pygram
from blib2to3.pgen2 import token
from blib2to3.pytree import NL, Leaf, Node, type_repr
def is_stub_body(node: LN) -> bool:
    """Return True if `node` is a simple statement containing an ellipsis."""
    if not isinstance(node, Node) or node.type != syms.simple_stmt:
        return False
    if len(node.children) != 2:
        return False
    child = node.children[0]
    return not child.prefix.strip() and child.type == syms.atom and (len(child.children) == 3) and all((leaf == Leaf(token.DOT, '.') for leaf in child.children))