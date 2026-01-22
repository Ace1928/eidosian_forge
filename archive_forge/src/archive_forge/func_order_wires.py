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
def order_wires(self, subset_wires):
    """Given some subset of device wires return a Wires object with the same wires;
        sorted according to the device wire map.

        Args:
            subset_wires (Wires): The subset of device wires (in any order).

        Raise:
            ValueError: Could not find some or all subset wires subset_wires in device wires device_wires.

        Return:
            ordered_wires (Wires): a new Wires object containing the re-ordered wires set
        """
    subset_lst = subset_wires.tolist()
    try:
        ordered_subset_lst = sorted(subset_lst, key=lambda label: self.wire_map[label])
    except KeyError as e:
        raise ValueError(f'Could not find some or all subset wires {subset_wires} in device wires {self.wires}') from e
    return Wires(ordered_subset_lst)