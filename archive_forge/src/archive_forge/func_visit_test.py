import re
import sys
from dataclasses import replace
from enum import Enum, auto
from functools import partial, wraps
from typing import Collection, Iterator, List, Optional, Set, Union, cast
from black.brackets import (
from black.comments import FMT_OFF, generate_comments, list_comments
from black.lines import (
from black.mode import Feature, Mode, Preview
from black.nodes import (
from black.numerics import normalize_numeric_literal
from black.strings import (
from black.trans import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def visit_test(self, node: Node) -> Iterator[Line]:
    """Visit an `x if y else z` test"""
    already_parenthesized = node.prev_sibling and node.prev_sibling.type == token.LPAR
    if not already_parenthesized:
        lpar = Leaf(token.LPAR, '')
        rpar = Leaf(token.RPAR, '')
        prefix = node.prefix
        node.prefix = ''
        lpar.prefix = prefix
        node.insert_child(0, lpar)
        node.append_child(rpar)
    yield from self.visit_default(node)