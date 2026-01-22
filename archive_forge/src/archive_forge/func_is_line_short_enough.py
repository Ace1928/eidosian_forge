import itertools
import math
from dataclasses import dataclass, field
from typing import (
from black.brackets import COMMA_PRIORITY, DOT_PRIORITY, BracketTracker
from black.mode import Mode, Preview
from black.nodes import (
from black.strings import str_width
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def is_line_short_enough(line: Line, *, mode: Mode, line_str: str='') -> bool:
    """For non-multiline strings, return True if `line` is no longer than `line_length`.
    For multiline strings, looks at the context around `line` to determine
    if it should be inlined or split up.
    Uses the provided `line_str` rendering, if any, otherwise computes a new one.
    """
    if not line_str:
        line_str = line_to_string(line)
    if Preview.multiline_string_handling not in mode:
        return str_width(line_str) <= mode.line_length and '\n' not in line_str and (not line.contains_standalone_comments())
    if line.contains_standalone_comments():
        return False
    if '\n' not in line_str:
        return str_width(line_str) <= mode.line_length
    first, *_, last = line_str.split('\n')
    if str_width(first) > mode.line_length or str_width(last) > mode.line_length:
        return False
    commas: List[int] = []
    multiline_string: Optional[Leaf] = None
    multiline_string_contexts: List[LN] = []
    max_level_to_update: Union[int, float] = math.inf
    for i, leaf in enumerate(line.leaves):
        if max_level_to_update == math.inf:
            had_comma: Optional[int] = None
            if leaf.bracket_depth + 1 > len(commas):
                commas.append(0)
            elif leaf.bracket_depth + 1 < len(commas):
                had_comma = commas.pop()
            if had_comma is not None and multiline_string is not None and (multiline_string.bracket_depth == leaf.bracket_depth + 1):
                max_level_to_update = leaf.bracket_depth
                if had_comma > 0:
                    return False
        if leaf.bracket_depth <= max_level_to_update and leaf.type == token.COMMA:
            ignore_ctxs: List[Optional[LN]] = [None]
            ignore_ctxs += multiline_string_contexts
            if (line.inside_brackets or leaf.bracket_depth > 0) and (i != len(line.leaves) - 1 or leaf.prev_sibling not in ignore_ctxs):
                commas[leaf.bracket_depth] += 1
        if max_level_to_update != math.inf:
            max_level_to_update = min(max_level_to_update, leaf.bracket_depth)
        if is_multiline_string(leaf):
            if len(multiline_string_contexts) > 0:
                return False
            multiline_string = leaf
            ctx: LN = leaf
            while str(ctx) in line_str:
                multiline_string_contexts.append(ctx)
                if ctx.parent is None:
                    break
                ctx = ctx.parent
    if len(multiline_string_contexts) == 0:
        return True
    return all((val == 0 for val in commas))