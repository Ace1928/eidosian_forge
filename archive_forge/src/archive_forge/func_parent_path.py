import math
from typing import (
import numpy as np
import sympy
from cirq import circuits, ops, protocols, value, study
from cirq._compat import cached_property, proper_repr
@property
def parent_path(self) -> Tuple[str, ...]:
    return self._parent_path