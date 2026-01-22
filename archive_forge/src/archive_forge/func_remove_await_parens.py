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
def remove_await_parens(node: Node) -> None:
    if node.children[0].type == token.AWAIT and len(node.children) > 1:
        if node.children[1].type == syms.atom and node.children[1].children[0].type == token.LPAR:
            if maybe_make_parens_invisible_in_atom(node.children[1], parent=node, remove_brackets_around_comma=True):
                wrap_in_parentheses(node, node.children[1], visible=False)
            opening_bracket = cast(Leaf, node.children[1].children[0])
            closing_bracket = cast(Leaf, node.children[1].children[-1])
            bracket_contents = node.children[1].children[1]
            if isinstance(bracket_contents, Node) and (bracket_contents.type != syms.power or bracket_contents.children[0].type == token.AWAIT or any((isinstance(child, Leaf) and child.type == token.DOUBLESTAR for child in bracket_contents.children))):
                ensure_visible(opening_bracket)
                ensure_visible(closing_bracket)