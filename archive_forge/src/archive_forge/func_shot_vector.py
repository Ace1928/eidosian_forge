import abc
import copy
import types
import warnings
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from functools import lru_cache
import numpy as np
import pennylane as qml
from pennylane.measurements import (
from pennylane.operation import Observable, Operation, Tensor, Operator, StatePrepBase
from pennylane.ops import Hamiltonian, Sum
from pennylane.tape import QuantumScript, QuantumTape, expand_tape_state_prep
from pennylane.wires import WireError, Wires
from pennylane.queuing import QueuingManager
@property
def shot_vector(self):
    """list[~pennylane.measurements.ShotCopies]: Returns the shot vector, a sparse
        representation of the shot sequence used by the device
        when evaluating QNodes.

        **Example**

        >>> dev = qml.device("default.qubit.legacy", wires=2, shots=[3, 1, 2, 2, 2, 2, 6, 1, 1, 5, 12, 10, 10])
        >>> dev.shots
        57
        >>> dev.shot_vector
        [ShotCopies(3 shots x 1),
         ShotCopies(1 shots x 1),
         ShotCopies(2 shots x 4),
         ShotCopies(6 shots x 1),
         ShotCopies(1 shots x 2),
         ShotCopies(5 shots x 1),
         ShotCopies(12 shots x 1),
         ShotCopies(10 shots x 2)]

        The sparse representation of the shot
        sequence is returned, where tuples indicate the number of times a shot
        integer is repeated.
        """
    return self._shot_vector