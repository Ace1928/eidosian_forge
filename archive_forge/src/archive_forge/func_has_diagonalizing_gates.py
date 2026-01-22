import abc
from typing import Callable, List
import copy
import pennylane as qml
from pennylane import math
from pennylane.operation import Operator, _UNSET_BATCH_SIZE
from pennylane.wires import Wires
@property
def has_diagonalizing_gates(self):
    if self.has_overlapping_wires:
        return self.has_matrix
    return all((op.has_diagonalizing_gates for op in self))