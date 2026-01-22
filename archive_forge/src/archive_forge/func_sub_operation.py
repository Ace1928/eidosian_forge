from typing import (
import numpy as np
from cirq import protocols, qis, value
from cirq.ops import (
from cirq.type_workarounds import NotImplementedType
@property
def sub_operation(self) -> 'cirq.Operation':
    return self._sub_operation