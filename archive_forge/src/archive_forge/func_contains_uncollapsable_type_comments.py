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
def contains_uncollapsable_type_comments(self) -> bool:
    ignored_ids = set()
    try:
        last_leaf = self.leaves[-1]
        ignored_ids.add(id(last_leaf))
        if last_leaf.type == token.COMMA or (last_leaf.type == token.RPAR and (not last_leaf.value)):
            last_leaf = self.leaves[-2]
            ignored_ids.add(id(last_leaf))
    except IndexError:
        return False
    comment_seen = False
    for leaf_id, comments in self.comments.items():
        for comment in comments:
            if is_type_comment(comment):
                if comment_seen or (not is_type_ignore_comment(comment) and leaf_id not in ignored_ids):
                    return True
            comment_seen = True
    return False