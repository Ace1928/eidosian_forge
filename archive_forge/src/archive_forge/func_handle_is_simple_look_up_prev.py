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
def handle_is_simple_look_up_prev(line: Line, index: int, disallowed: Set[int]) -> bool:
    """
    Handling the determination of is_simple_lookup for the lines prior to the doublestar
    token. This is required because of the need to isolate the chained expression
    to determine the bracket or parenthesis belong to the single expression.
    """
    contains_disallowed = False
    chain = []
    while 0 <= index < len(line.leaves):
        current = line.leaves[index]
        chain.append(current)
        if not contains_disallowed and current.type in disallowed:
            contains_disallowed = True
        if not is_expression_chained(chain):
            return not contains_disallowed
        index -= 1
    return True