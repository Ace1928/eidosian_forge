import sys
from typing import (
from mypy_extensions import mypyc_attr
from black.cache import CACHE_DIR
from black.mode import Mode, Preview
from black.strings import get_string_prefix, has_triple_quotes
from blib2to3 import pygram
from blib2to3.pgen2 import token
from blib2to3.pytree import NL, Leaf, Node, type_repr
def wrap_in_parentheses(parent: Node, child: LN, *, visible: bool=True) -> None:
    """Wrap `child` in parentheses.

    This replaces `child` with an atom holding the parentheses and the old
    child.  That requires moving the prefix.

    If `visible` is False, the leaves will be valueless (and thus invisible).
    """
    lpar = Leaf(token.LPAR, '(' if visible else '')
    rpar = Leaf(token.RPAR, ')' if visible else '')
    prefix = child.prefix
    child.prefix = ''
    index = child.remove() or 0
    new_child = Node(syms.atom, [lpar, child, rpar])
    new_child.prefix = prefix
    parent.insert_child(index, new_child)