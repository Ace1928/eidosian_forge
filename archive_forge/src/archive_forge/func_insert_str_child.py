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
def insert_str_child(child: LN) -> None:
    nonlocal string_child_idx
    assert string_parent is not None
    assert string_child_idx is not None
    string_parent.insert_child(string_child_idx, child)
    string_child_idx += 1