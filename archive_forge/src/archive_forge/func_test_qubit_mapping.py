import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
def test_qubit_mapping():
    q0, q1 = cirq.LineQubit.range(2)
    op = cirq.X(q0).with_classical_controls('a')
    assert op.with_qubits(q1).qubits == (q1,)