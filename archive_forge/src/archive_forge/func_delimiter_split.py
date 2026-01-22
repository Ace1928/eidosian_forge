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
@dont_increase_indentation
def delimiter_split(line: Line, features: Collection[Feature], mode: Mode) -> Iterator[Line]:
    """Split according to delimiters of the highest priority.

    If the appropriate Features are given, the split will add trailing commas
    also in function signatures and calls that contain `*` and `**`.
    """
    if len(line.leaves) == 0:
        raise CannotSplit('Line empty') from None
    last_leaf = line.leaves[-1]
    bt = line.bracket_tracker
    try:
        delimiter_priority = bt.max_delimiter_priority(exclude={id(last_leaf)})
    except ValueError:
        raise CannotSplit('No delimiters found') from None
    if delimiter_priority == DOT_PRIORITY and bt.delimiter_count_with_priority(delimiter_priority) == 1:
        raise CannotSplit('Splitting a single attribute from its owner looks wrong')
    current_line = Line(mode=line.mode, depth=line.depth, inside_brackets=line.inside_brackets)
    lowest_depth = sys.maxsize
    trailing_comma_safe = True

    def append_to_line(leaf: Leaf) -> Iterator[Line]:
        """Append `leaf` to current line or to new line if appending impossible."""
        nonlocal current_line
        try:
            current_line.append_safe(leaf, preformatted=True)
        except ValueError:
            yield current_line
            current_line = Line(mode=line.mode, depth=line.depth, inside_brackets=line.inside_brackets)
            current_line.append(leaf)

    def append_comments(leaf: Leaf) -> Iterator[Line]:
        for comment_after in line.comments_after(leaf):
            yield from append_to_line(comment_after)
    last_non_comment_leaf = _get_last_non_comment_leaf(line)
    for leaf_idx, leaf in enumerate(line.leaves):
        yield from append_to_line(leaf)
        previous_priority = leaf_idx > 0 and bt.delimiters.get(id(line.leaves[leaf_idx - 1]))
        if previous_priority != delimiter_priority or delimiter_priority in MIGRATE_COMMENT_DELIMITERS:
            yield from append_comments(leaf)
        lowest_depth = min(lowest_depth, leaf.bracket_depth)
        if trailing_comma_safe and leaf.bracket_depth == lowest_depth:
            trailing_comma_safe = _can_add_trailing_comma(leaf, features)
        if last_leaf.type == STANDALONE_COMMENT and leaf_idx == last_non_comment_leaf:
            current_line = _safe_add_trailing_comma(trailing_comma_safe, delimiter_priority, current_line)
        leaf_priority = bt.delimiters.get(id(leaf))
        if leaf_priority == delimiter_priority:
            if leaf_idx + 1 < len(line.leaves) and delimiter_priority not in MIGRATE_COMMENT_DELIMITERS:
                yield from append_comments(line.leaves[leaf_idx + 1])
            yield current_line
            current_line = Line(mode=line.mode, depth=line.depth, inside_brackets=line.inside_brackets)
    if current_line:
        current_line = _safe_add_trailing_comma(trailing_comma_safe, delimiter_priority, current_line)
        yield current_line