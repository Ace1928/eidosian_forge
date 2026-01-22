from __future__ import annotations
import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.parameterexpression import ParameterValueType
def rzx_xz(theta: ParameterValueType | None=None):
    """Template for CX - RXGate - CX."""
    if theta is None:
        theta = Parameter('ϴ')
    qc = QuantumCircuit(2)
    qc.cx(1, 0)
    qc.rx(theta, 1)
    qc.cx(1, 0)
    qc.rz(np.pi / 2, 0)
    qc.rx(np.pi / 2, 0)
    qc.rz(np.pi / 2, 0)
    qc.rzx(-1 * theta, 0, 1)
    qc.rz(np.pi / 2, 0)
    qc.rx(np.pi / 2, 0)
    qc.rz(np.pi / 2, 0)
    return qc