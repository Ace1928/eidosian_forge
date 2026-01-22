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
def visit_simple_stmt(self, node: Node) -> Iterator[Line]:
    """Visit a statement without nested statements."""
    prev_type: Optional[int] = None
    for child in node.children:
        if (prev_type is None or prev_type == token.SEMI) and is_arith_like(child):
            wrap_in_parentheses(node, child, visible=False)
        prev_type = child.type
    if node.parent and node.parent.type in STATEMENT:
        if is_parent_function_or_class(node) and is_stub_body(node):
            yield from self.visit_default(node)
        else:
            yield from self.line(+1)
            yield from self.visit_default(node)
            yield from self.line(-1)
    else:
        if not node.parent or not is_stub_suite(node.parent):
            yield from self.line()
        yield from self.visit_default(node)