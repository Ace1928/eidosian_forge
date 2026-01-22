import re
import os
import numpy as np
import pytest
import cirq
from cirq.circuits.qasm_output import QasmTwoQubitGate, QasmUGate
from cirq.testing import consistent_qasm as cq
def test_u_gate_eq():
    gate = QasmUGate(0.1, 0.2, 0.3)
    gate2 = QasmUGate(0.1, 0.2, 0.3)
    cirq.approx_eq(gate, gate2, atol=1e-16)
    gate3 = QasmUGate(0.1, 0.2, 0.4)
    gate4 = QasmUGate(0.1, 0.2, 2.4)
    cirq.approx_eq(gate4, gate3, atol=1e-16)