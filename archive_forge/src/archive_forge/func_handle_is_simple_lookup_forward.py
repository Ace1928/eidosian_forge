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
def handle_is_simple_lookup_forward(line: Line, index: int, disallowed: Set[int]) -> bool:
    """
    Handling decision is_simple_lookup for the lines behind the doublestar token.
    This function is simplified to keep consistent with the prior logic and the forward
    case are more straightforward and do not need to care about chained expressions.
    """
    while 0 <= index < len(line.leaves):
        current = line.leaves[index]
        if current.type in disallowed:
            return False
        if current.type not in {token.NAME, token.DOT} or (current.type == token.NAME and current.value == 'for'):
            return True
        index += 1
    return True