from typing import (
from blib2to3.pgen2.grammar import Grammar
import sys
from io import StringIO
def invalidate_sibling_maps(self) -> None:
    self.prev_sibling_map: Optional[Dict[int, Optional[NL]]] = None
    self.next_sibling_map: Optional[Dict[int, Optional[NL]]] = None