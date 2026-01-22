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
def test_unexpected_end_of_file():
    qasm = 'OPENQASM 2.0;\n              include "qelib1.inc";\n              creg\n           '
    parser = QasmParser()
    with pytest.raises(QasmException, match='Unexpected end of file'):
        parser.parse(qasm)