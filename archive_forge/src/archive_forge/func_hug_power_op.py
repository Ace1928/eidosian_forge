import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import (
from mypy_extensions import trait
from black.comments import contains_pragma_comment
from black.lines import Line, append_leaves
from black.mode import Feature, Mode, Preview
from black.nodes import (
from black.rusty import Err, Ok, Result
from black.strings import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def hug_power_op(line: Line, features: Collection[Feature], mode: Mode) -> Iterator[Line]:
    """A transformer which normalizes spacing around power operators."""
    for leaf in line.leaves:
        if leaf.type == token.DOUBLESTAR:
            break
    else:
        raise CannotTransform('No doublestar token was found in the line.')

    def is_simple_lookup(index: int, kind: Literal[1, -1]) -> bool:
        if Preview.is_simple_lookup_for_doublestar_expression not in mode:
            return original_is_simple_lookup_func(line, index, kind)
        elif kind == -1:
            return handle_is_simple_look_up_prev(line, index, {token.RPAR, token.RSQB})
        else:
            return handle_is_simple_lookup_forward(line, index, {token.LPAR, token.LSQB})

    def is_simple_operand(index: int, kind: Literal[1, -1]) -> bool:
        start = line.leaves[index]
        if start.type in {token.NAME, token.NUMBER}:
            return is_simple_lookup(index, kind)
        if start.type in {token.PLUS, token.MINUS, token.TILDE}:
            if line.leaves[index + 1].type in {token.NAME, token.NUMBER}:
                return is_simple_lookup(index + 1, kind=1)
        return False
    new_line = line.clone()
    should_hug = False
    for idx, leaf in enumerate(line.leaves):
        new_leaf = leaf.clone()
        if should_hug:
            new_leaf.prefix = ''
            should_hug = False
        should_hug = 0 < idx < len(line.leaves) - 1 and leaf.type == token.DOUBLESTAR and is_simple_operand(idx - 1, kind=-1) and (line.leaves[idx - 1].value != 'lambda') and is_simple_operand(idx + 1, kind=1)
        if should_hug:
            new_leaf.prefix = ''
        new_line.append(new_leaf, preformatted=True)
        for comment_leaf in line.comments_after(leaf):
            new_line.append(comment_leaf, preformatted=True)
    yield new_line