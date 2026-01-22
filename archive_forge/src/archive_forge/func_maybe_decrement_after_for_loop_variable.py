from dataclasses import dataclass, field
from typing import Dict, Final, Iterable, List, Optional, Sequence, Set, Tuple, Union
from black.nodes import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def maybe_decrement_after_for_loop_variable(self, leaf: Leaf) -> bool:
    """See `maybe_increment_for_loop_variable` above for explanation."""
    if self._for_loop_depths and self._for_loop_depths[-1] == self.depth and (leaf.type == token.NAME) and (leaf.value == 'in'):
        self.depth -= 1
        self._for_loop_depths.pop()
        return True
    return False