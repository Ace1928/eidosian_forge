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
def bracket_split_build_line(leaves: List[Leaf], original: Line, opening_bracket: Leaf, *, component: _BracketSplitComponent) -> Line:
    """Return a new line with given `leaves` and respective comments from `original`.

    If it's the head component, brackets will be tracked so trailing commas are
    respected.

    If it's the body component, the result line is one-indented inside brackets and as
    such has its first leaf's prefix normalized and a trailing comma added when
    expected.
    """
    result = Line(mode=original.mode, depth=original.depth)
    if component is _BracketSplitComponent.body:
        result.inside_brackets = True
        result.depth += 1
        if leaves:
            no_commas = original.is_def and opening_bracket.value == '(' and (not any((leaf.type == token.COMMA and (Preview.typed_params_trailing_comma not in original.mode or not is_part_of_annotation(leaf)) for leaf in leaves))) and (get_annotation_type(leaves[0]) != 'return') and (not (leaves[0].parent and leaves[0].parent.next_sibling and (leaves[0].parent.next_sibling.type == token.VBAR)))
            if original.is_import or no_commas:
                for i in range(len(leaves) - 1, -1, -1):
                    if leaves[i].type == STANDALONE_COMMENT:
                        continue
                    if leaves[i].type != token.COMMA:
                        new_comma = Leaf(token.COMMA, ',')
                        leaves.insert(i + 1, new_comma)
                    break
    leaves_to_track: Set[LeafID] = set()
    if component is _BracketSplitComponent.head:
        leaves_to_track = get_leaves_inside_matching_brackets(leaves)
    for leaf in leaves:
        result.append(leaf, preformatted=True, track_bracket=id(leaf) in leaves_to_track)
        for comment_after in original.comments_after(leaf):
            result.append(comment_after, preformatted=True)
    if component is _BracketSplitComponent.body and should_split_line(result, opening_bracket):
        result.should_split_rhs = True
    return result