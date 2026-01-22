from typing import AbstractSet, Union, Any, Optional, Tuple, TYPE_CHECKING, Dict
import numpy as np
from cirq import protocols, value
from cirq.ops import raw_types
from cirq.type_workarounds import NotImplementedType
def parallel_gate_op(gate: 'cirq.Gate', *targets: 'cirq.Qid') -> 'cirq.Operation':
    """Constructs a ParallelGate using gate and applies to all given qubits

    Args:
        gate: The gate to apply
        *targets: The qubits on which the ParallelGate should be applied.

    Returns:
        ParallelGate(gate, len(targets)).on(*targets)

    """
    return ParallelGate(gate, len(targets)).on(*targets)