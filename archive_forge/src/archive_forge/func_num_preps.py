import contextlib
import copy
from collections import Counter
from typing import List, Union, Optional, Sequence
import pennylane as qml
from pennylane.measurements import (
from pennylane.typing import TensorLike
from pennylane.operation import Observable, Operator, Operation, _UNSET_BATCH_SIZE
from pennylane.pytrees import register_pytree
from pennylane.queuing import AnnotatedQueue, process_queue
from pennylane.wires import Wires
@property
def num_preps(self) -> int:
    """Returns the index of the first operator that is not an StatePrepBase operator."""
    idx = 0
    num_ops = len(self.operations)
    while idx < num_ops and isinstance(self.operations[idx], qml.operation.StatePrepBase):
        idx += 1
    return idx