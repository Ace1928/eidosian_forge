from dataclasses import dataclass, field
from typing import Dict, Final, Iterable, List, Optional, Sequence, Set, Tuple, Union
from black.nodes import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def maybe_increment_for_loop_variable(self, leaf: Leaf) -> bool:
    """In a for loop, or comprehension, the variables are often unpacks.

        To avoid splitting on the comma in this situation, increase the depth of
        tokens between `for` and `in`.
        """
    if leaf.type == token.NAME and leaf.value == 'for':
        self.depth += 1
        self._for_loop_depths.append(self.depth)
        return True
    return False