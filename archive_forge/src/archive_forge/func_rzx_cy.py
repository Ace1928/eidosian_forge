from __future__ import annotations
import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.parameterexpression import ParameterValueType
def rzx_cy(theta: ParameterValueType | None=None):
    """Template for CX - RYGate - CX."""
    if theta is None:
        theta = Parameter('ϴ')
    circ = QuantumCircuit(2)
    circ.cx(0, 1)
    circ.ry(theta, 1)
    circ.cx(0, 1)
    circ.ry(-1 * theta, 1)
    circ.rz(-np.pi / 2, 1)
    circ.rx(theta, 1)
    circ.rzx(-1 * theta, 0, 1)
    circ.rz(np.pi / 2, 1)
    return circ