import abc
from typing import Callable, List
import copy
import pennylane as qml
from pennylane import math
from pennylane.operation import Operator, _UNSET_BATCH_SIZE
from pennylane.wires import Wires
@property
def num_wires(self):
    """Number of wires the operator acts on."""
    return len(self.wires)