from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
def test_inverse_composite_diagram_info():

    class Gate(cirq.Gate):

        def _decompose_(self, qubits):
            return cirq.S.on(qubits[0])

        def num_qubits(self) -> int:
            return 1
    c = cirq.inverse(Gate())
    assert cirq.circuit_diagram_info(c, default=None) is None

    class Gate2(cirq.Gate):

        def _decompose_(self, qubits):
            return cirq.S.on(qubits[0])

        def num_qubits(self) -> int:
            return 1

        def _circuit_diagram_info_(self, args):
            return 's!'
    c = cirq.inverse(Gate2())
    assert cirq.circuit_diagram_info(c) == cirq.CircuitDiagramInfo(wire_symbols=('s!',), exponent=-1)