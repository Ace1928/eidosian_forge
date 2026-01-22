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
@pytest.mark.parametrize('qasm', ['include "qelib1.inc";', '', 'qreg q[3];'])
def test_error_not_starting_with_format(qasm: str):
    parser = QasmParser()
    with pytest.raises(QasmException, match="Missing 'OPENQASM 2.0;' statement"):
        parser.parse(qasm)