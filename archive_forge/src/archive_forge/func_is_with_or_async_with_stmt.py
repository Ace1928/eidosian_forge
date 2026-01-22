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
def is_with_or_async_with_stmt(self) -> bool:
    """Is this a with_stmt line?"""
    return bool(self) and is_with_or_async_with_stmt(self.leaves[0])