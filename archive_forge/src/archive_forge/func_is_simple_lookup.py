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
def is_simple_lookup(index: int, kind: Literal[1, -1]) -> bool:
    if Preview.is_simple_lookup_for_doublestar_expression not in mode:
        return original_is_simple_lookup_func(line, index, kind)
    elif kind == -1:
        return handle_is_simple_look_up_prev(line, index, {token.RPAR, token.RSQB})
    else:
        return handle_is_simple_lookup_forward(line, index, {token.LPAR, token.LSQB})