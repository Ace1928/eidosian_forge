from __future__ import annotations
import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.parameterexpression import ParameterValueType
def rzx_zz2(theta: ParameterValueType | None=None):
    """Template for CX - RZGate - CX."""
    if theta is None:
        theta = Parameter('ϴ')
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.p(theta, 1)
    qc.cx(0, 1)
    qc.p(-1 * theta, 1)
    qc.rz(np.pi / 2, 1)
    qc.rx(np.pi / 2, 1)
    qc.rz(np.pi / 2, 1)
    qc.rx(theta, 1)
    qc.rzx(-1 * theta, 0, 1)
    qc.rz(np.pi / 2, 1)
    qc.rx(np.pi / 2, 1)
    qc.rz(np.pi / 2, 1)
    return qc