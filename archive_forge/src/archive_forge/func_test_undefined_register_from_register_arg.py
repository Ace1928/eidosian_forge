from typing import Callable
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing as ct
from cirq import Circuit
from cirq.circuits.qasm_output import QasmUGate
from cirq.contrib.qasm_import import QasmException
from cirq.contrib.qasm_import._parser import QasmParser
def test_undefined_register_from_register_arg():
    qasm = 'OPENQASM 2.0;\n            qreg q[2];\n            qreg q2[2];\n            CX q1, q2;\n       '
    parser = QasmParser()
    with pytest.raises(QasmException, match='Undefined.*register.*q.*'):
        parser.parse(qasm)