import math
from typing import (
import numpy as np
import sympy
from cirq import circuits, ops, protocols, value, study
from cirq._compat import cached_property, proper_repr
@property
def repetition_ids(self) -> Optional[Sequence[str]]:
    return self._repetition_ids