from dataclasses import dataclass, field
from typing import Dict, Final, Iterable, List, Optional, Sequence, Set, Tuple, Union
from black.nodes import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def get_leaves_inside_matching_brackets(leaves: Sequence[Leaf]) -> Set[LeafID]:
    """Return leaves that are inside matching brackets.

    The input `leaves` can have non-matching brackets at the head or tail parts.
    Matching brackets are included.
    """
    try:
        start_index = next((i for i, l in enumerate(leaves) if l.type in OPENING_BRACKETS))
    except StopIteration:
        return set()
    bracket_stack = []
    ids = set()
    for i in range(start_index, len(leaves)):
        leaf = leaves[i]
        if leaf.type in OPENING_BRACKETS:
            bracket_stack.append((BRACKET[leaf.type], i))
        if leaf.type in CLOSING_BRACKETS:
            if bracket_stack and leaf.type == bracket_stack[-1][0]:
                _, start = bracket_stack.pop()
                for j in range(start, i + 1):
                    ids.add(id(leaves[j]))
            else:
                break
    return ids