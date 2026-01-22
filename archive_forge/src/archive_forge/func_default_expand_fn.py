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
def default_expand_fn(self, circuit, max_expansion=10):
    """Method for expanding or decomposing an input circuit.
        This method should be overwritten if custom expansion logic is
        required.

        By default, this method expands the tape if:

        - state preparation operations are called mid-circuit,
        - nested tapes are present,
        - any operations are not supported on the device, or
        - multiple observables are measured on the same wire.

        Args:
            circuit (.QuantumTape): the circuit to expand.
            max_expansion (int): The number of times the circuit should be
                expanded. Expansion occurs when an operation or measurement is not
                supported, and results in a gate decomposition. If any operations
                in the decomposition remain unsupported by the device, another
                expansion occurs.

        Returns:
            .QuantumTape: The expanded/decomposed circuit, such that the device
            will natively support all operations.
        """
    if max_expansion == 0:
        return circuit
    expand_state_prep = any((isinstance(op, StatePrepBase) for op in circuit.operations[1:]))
    if expand_state_prep:
        circuit = expand_tape_state_prep(circuit)
    comp_basis_sampled_multi_measure = len(circuit.measurements) > 1 and circuit.samples_computational_basis
    obs_on_same_wire = len(circuit._obs_sharing_wires) > 0 or comp_basis_sampled_multi_measure
    obs_on_same_wire &= not any((isinstance(o, qml.Hamiltonian) for o in circuit._obs_sharing_wires))
    ops_not_supported = not all((self.stopping_condition(op) for op in circuit.operations))
    if obs_on_same_wire:
        circuit = circuit.expand(depth=max_expansion, stop_at=self.stopping_condition)
    elif ops_not_supported:
        circuit = _local_tape_expand(circuit, depth=max_expansion, stop_at=self.stopping_condition)
        circuit._update()
    return circuit