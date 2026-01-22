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
def visit_funcdef(self, node: Node) -> Iterator[Line]:
    """Visit function definition."""
    yield from self.line()
    is_return_annotation = False
    for child in node.children:
        if child.type == token.RARROW:
            is_return_annotation = True
        elif is_return_annotation:
            if child.type == syms.atom and child.children[0].type == token.LPAR:
                if maybe_make_parens_invisible_in_atom(child, parent=node, remove_brackets_around_comma=False):
                    wrap_in_parentheses(node, child, visible=False)
            else:
                wrap_in_parentheses(node, child, visible=False)
            is_return_annotation = False
    for child in node.children:
        yield from self.visit(child)