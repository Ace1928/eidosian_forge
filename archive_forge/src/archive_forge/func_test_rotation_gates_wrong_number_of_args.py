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
@pytest.mark.parametrize('qasm_gate', [g[0] for g in rotation_gates])
def test_rotation_gates_wrong_number_of_args(qasm_gate: str):
    qasm = f'\n     OPENQASM 2.0;\n     include "qelib1.inc";             \n     qreg q[2];     \n     {qasm_gate}(pi) q[0], q[1];     \n'
    parser = QasmParser()
    with pytest.raises(QasmException, match=f'.*{qasm_gate}.* takes 1.*got.*2.*line 5'):
        parser.parse(qasm)