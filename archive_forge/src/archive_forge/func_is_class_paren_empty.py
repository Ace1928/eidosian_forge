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
def is_class_paren_empty(self) -> bool:
    """Is this a class with no base classes but using parentheses?

        Those are unnecessary and should be removed.
        """
    return bool(self) and len(self.leaves) == 4 and self.is_class and (self.leaves[2].type == token.LPAR) and (self.leaves[2].value == '(') and (self.leaves[3].type == token.RPAR) and (self.leaves[3].value == ')')