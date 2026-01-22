import sys
from typing import (
from mypy_extensions import mypyc_attr
from black.cache import CACHE_DIR
from black.mode import Mode, Preview
from black.strings import get_string_prefix, has_triple_quotes
from blib2to3 import pygram
from blib2to3.pgen2 import token
from blib2to3.pytree import NL, Leaf, Node, type_repr
def is_simple_decorator_trailer(node: LN, last: bool=False) -> bool:
    """Return True iff `node` is a trailer valid in a simple decorator"""
    return node.type == syms.trailer and (len(node.children) == 2 and node.children[0].type == token.DOT and (node.children[1].type == token.NAME) or (last and len(node.children) == 2 and (node.children[0].type == token.LPAR) and (node.children[1].type == token.RPAR)) or (last and len(node.children) == 3 and (node.children[0].type == token.LPAR) and (node.children[2].type == token.RPAR)))