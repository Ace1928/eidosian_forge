from typing import (
import numpy as np
from cirq import protocols, value, _import
from cirq.ops import (
from cirq.type_workarounds import NotImplementedType
@property
def control_qid_shape(self) -> Tuple[int, ...]:
    return self._control_qid_shape