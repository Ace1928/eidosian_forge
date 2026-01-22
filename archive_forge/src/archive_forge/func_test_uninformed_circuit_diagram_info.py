import itertools
import re
from typing import cast, Tuple, Union
import numpy as np
import pytest
import sympy
import cirq
from cirq import protocols
from cirq.type_workarounds import NotImplementedType
def test_uninformed_circuit_diagram_info():
    qbits = cirq.LineQubit.range(3)
    mock_gate = MockGate()
    c_op = cirq.ControlledOperation(qbits[:1], mock_gate(*qbits[1:]))
    args = protocols.CircuitDiagramInfoArgs.UNINFORMED_DEFAULT
    assert cirq.circuit_diagram_info(c_op, args) == cirq.CircuitDiagramInfo(wire_symbols=('@', 'M1', 'M2'), exponent=1, connected=True, exponent_qubit_index=1)
    assert mock_gate.captured_diagram_args == args