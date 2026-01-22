import re
import os
import numpy as np
import pytest
import cirq
from cirq.circuits.qasm_output import QasmTwoQubitGate, QasmUGate
from cirq.testing import consistent_qasm as cq
def test_unsupported_operation():
    q0, = _make_qubits(1)

    class UnsupportedOperation(cirq.Operation):
        qubits = (q0,)
        with_qubits = NotImplemented
    output = cirq.QasmOutput((UnsupportedOperation(),), (q0,))
    with pytest.raises(ValueError):
        _ = str(output)