import itertools
from typing import (
from typing_extensions import Self
import numpy as np
from cirq import protocols, ops, qis, _compat
from cirq._import import LazyLoader
from cirq.ops import raw_types, op_tree
from cirq.protocols import circuit_diagram_info_protocol
from cirq.type_workarounds import NotImplementedType
def kraus_tensors(op: 'cirq.Operation') -> Sequence[np.ndarray]:
    return tuple((np.reshape(k, (2, 2) * len(op.qubits)) for k in protocols.kraus(op)))