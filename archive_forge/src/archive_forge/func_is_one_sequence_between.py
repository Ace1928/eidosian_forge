import sys
from typing import (
from mypy_extensions import mypyc_attr
from black.cache import CACHE_DIR
from black.mode import Mode, Preview
from black.strings import get_string_prefix, has_triple_quotes
from blib2to3 import pygram
from blib2to3.pgen2 import token
from blib2to3.pytree import NL, Leaf, Node, type_repr
def is_one_sequence_between(opening: Leaf, closing: Leaf, leaves: List[Leaf], brackets: Tuple[int, int]=(token.LPAR, token.RPAR)) -> bool:
    """Return True if content between `opening` and `closing` is a one-sequence."""
    if (opening.type, closing.type) != brackets:
        return False
    depth = closing.bracket_depth + 1
    for _opening_index, leaf in enumerate(leaves):
        if leaf is opening:
            break
    else:
        raise LookupError('Opening paren not found in `leaves`')
    commas = 0
    _opening_index += 1
    for leaf in leaves[_opening_index:]:
        if leaf is closing:
            break
        bracket_depth = leaf.bracket_depth
        if bracket_depth == depth and leaf.type == token.COMMA:
            commas += 1
            if leaf.parent and leaf.parent.type in {syms.arglist, syms.typedargslist}:
                commas += 1
                break
    return commas < 2