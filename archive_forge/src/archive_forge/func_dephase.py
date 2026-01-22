import itertools
from collections import defaultdict
from typing import Any, cast, Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import numpy as np
from cirq import linalg, ops, protocols, value
from cirq.linalg import transformations
from cirq.transformers import transformer_api, transformer_primitives
from cirq.transformers.synchronize_terminal_measurements import find_terminal_measurements
def dephase(op: 'cirq.Operation', _) -> 'cirq.OP_TREE':
    gate = op.gate
    if isinstance(gate, ops.MeasurementGate):
        key = value.MeasurementKey.parse_serialized(gate.key)
        return ops.KrausChannel.from_channel(ops.phase_damp(1), key=key).on_each(op.qubits)
    elif isinstance(op, ops.ClassicallyControlledOperation):
        raise ValueError('Use cirq.defer_measurements first to remove classical controls.')
    return op