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
def visit_dictsetmaker(self, node: Node) -> Iterator[Line]:
    if Preview.wrap_long_dict_values_in_parens in self.mode:
        for i, child in enumerate(node.children):
            if i == 0:
                continue
            if node.children[i - 1].type == token.COLON:
                if child.type == syms.atom and child.children[0].type in OPENING_BRACKETS and (not is_walrus_assignment(child)):
                    maybe_make_parens_invisible_in_atom(child, parent=node, remove_brackets_around_comma=False)
                else:
                    wrap_in_parentheses(node, child, visible=False)
    yield from self.visit_default(node)