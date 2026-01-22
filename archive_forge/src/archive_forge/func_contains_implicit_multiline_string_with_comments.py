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
def contains_implicit_multiline_string_with_comments(self) -> bool:
    """Chck if we have an implicit multiline string with comments on the line"""
    for leaf_type, leaf_group_iterator in itertools.groupby(self.leaves, lambda leaf: leaf.type):
        if leaf_type != token.STRING:
            continue
        leaf_list = list(leaf_group_iterator)
        if len(leaf_list) == 1:
            continue
        for leaf in leaf_list:
            if self.comments_after(leaf):
                return True
    return False