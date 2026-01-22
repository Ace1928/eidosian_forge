import sys
from typing import (
from mypy_extensions import mypyc_attr
from black.cache import CACHE_DIR
from black.mode import Mode, Preview
from black.strings import get_string_prefix, has_triple_quotes
from blib2to3 import pygram
from blib2to3.pgen2 import token
from blib2to3.pytree import NL, Leaf, Node, type_repr
def is_type_ignore_comment(leaf: Leaf) -> bool:
    """Return True if the given leaf is a type comment with ignore annotation."""
    t = leaf.type
    v = leaf.value
    return t in {token.COMMENT, STANDALONE_COMMENT} and is_type_ignore_comment_string(v)