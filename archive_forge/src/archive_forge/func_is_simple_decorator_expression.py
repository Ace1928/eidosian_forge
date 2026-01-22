import sys
from typing import (
from mypy_extensions import mypyc_attr
from black.cache import CACHE_DIR
from black.mode import Mode, Preview
from black.strings import get_string_prefix, has_triple_quotes
from blib2to3 import pygram
from blib2to3.pgen2 import token
from blib2to3.pytree import NL, Leaf, Node, type_repr
def is_simple_decorator_expression(node: LN) -> bool:
    """Return True iff `node` could be a 'dotted name' decorator

    This function takes the node of the 'namedexpr_test' of the new decorator
    grammar and test if it would be valid under the old decorator grammar.

    The old grammar was: decorator: @ dotted_name [arguments] NEWLINE
    The new grammar is : decorator: @ namedexpr_test NEWLINE
    """
    if node.type == token.NAME:
        return True
    if node.type == syms.power:
        if node.children:
            return node.children[0].type == token.NAME and all(map(is_simple_decorator_trailer, node.children[1:-1])) and (len(node.children) < 2 or is_simple_decorator_trailer(node.children[-1], last=True))
    return False