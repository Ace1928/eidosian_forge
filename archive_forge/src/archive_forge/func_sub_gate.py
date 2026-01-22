from typing import (
import numpy as np
from cirq import protocols, value, _import
from cirq.ops import (
from cirq.type_workarounds import NotImplementedType
@property
def sub_gate(self) -> 'cirq.Gate':
    return self._sub_gate