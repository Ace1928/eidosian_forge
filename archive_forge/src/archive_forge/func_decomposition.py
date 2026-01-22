import copy
from typing import Union
from scipy.linalg import fractional_matrix_power
import pennylane as qml
from pennylane import math as qmlmath
from pennylane.operation import (
from pennylane.ops.identity import Identity
from pennylane.queuing import QueuingManager, apply
from .symbolicop import ScalarSymbolicOp
def decomposition(self):
    try:
        return self.base.pow(self.z)
    except PowUndefinedError as e:
        if isinstance(self.z, int) and self.z > 0:
            if QueuingManager.recording():
                return [apply(self.base) for _ in range(self.z)]
            return [copy.copy(self.base) for _ in range(self.z)]
        raise DecompositionUndefinedError from e