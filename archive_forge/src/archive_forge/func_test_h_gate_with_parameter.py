import re
import os
import numpy as np
import pytest
import cirq
from cirq.circuits.qasm_output import QasmTwoQubitGate, QasmUGate
from cirq.testing import consistent_qasm as cq
def test_h_gate_with_parameter():
    q0, = _make_qubits(1)
    output = cirq.QasmOutput((cirq.H(q0) ** 0.25,), (q0,))
    assert str(output) == 'OPENQASM 2.0;\ninclude "qelib1.inc";\n\n\n// Qubits: [q0]\nqreg q[1];\n\n\n// Gate: H**0.25\nry(pi*0.25) q[0];\nrx(pi*0.25) q[0];\nry(pi*-0.25) q[0];\n'