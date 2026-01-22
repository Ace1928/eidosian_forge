import math
from typing import (
import numpy as np
import sympy
from cirq import circuits, ops, protocols, value, study
from cirq._compat import cached_property, proper_repr
@property
def measurement_key_map(self) -> Mapping[str, str]:
    return self._measurement_key_map