from itertools import combinations
from string import ascii_lowercase
from typing import Sequence, Dict, Tuple
import numpy as np
import pytest
import cirq
import cirq.testing as ct
import cirq.contrib.acquaintance as cca
def test_acquaintance_operation():
    n = 5
    physical_qubits = tuple(cirq.LineQubit.range(n))
    logical_qubits = tuple((cirq.NamedQubit(s) for s in ascii_lowercase[:n]))
    int_indices = tuple(range(n))
    with pytest.raises(ValueError):
        cca.AcquaintanceOperation(physical_qubits[:3], int_indices[:4])
    for logical_indices in (logical_qubits, int_indices):
        op = cca.AcquaintanceOperation(physical_qubits, logical_indices)
        assert op.logical_indices == logical_indices
        assert op.qubits == physical_qubits
        wire_symbols = tuple((f'({i})' for i in logical_indices))
        assert cirq.circuit_diagram_info(op) == cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)