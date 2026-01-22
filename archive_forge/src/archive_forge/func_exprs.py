from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.model import (
from .types_base import Binding, CType, Expr
from torchgen.api import (
def exprs(self) -> List[Expr]:
    return [Expr(a.name, a.nctype) for a in self.arguments()]