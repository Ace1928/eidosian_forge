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
@property
def is_chained_assignment(self) -> bool:
    """Is the line a chained assignment"""
    return [leaf.type for leaf in self.leaves].count(token.EQUAL) > 1