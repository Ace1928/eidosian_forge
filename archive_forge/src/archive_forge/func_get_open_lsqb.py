from dataclasses import dataclass, field
from typing import Dict, Final, Iterable, List, Optional, Sequence, Set, Tuple, Union
from black.nodes import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def get_open_lsqb(self) -> Optional[Leaf]:
    """Return the most recent opening square bracket (if any)."""
    return self.bracket_match.get((self.depth - 1, token.RSQB))