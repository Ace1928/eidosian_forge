import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires
def u1_ex_gate(phi, theta, wires=None):
    """Appends the two-qubit exchange gate :math:`U_{1,\\mathrm{ex}}` proposed
    in `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_ to build
    a hardware-efficient particle-conserving VQE ansatz for quantum chemistry
    simulations.

    Args:
        phi (float): angle entering the unitary :math:`U_A(\\phi)`
        theta (float): angle entering the rotation :math:`R(0, 2\\theta, 0)`
        wires (list[Iterable]): the two wires ``n`` and ``m`` the circuit acts on

    Returns:
        list[.Operator]: sequence of operators defined by this function
    """
    op_list = []
    op_list.extend(decompose_ua(phi, wires=wires))
    op_list.append(qml.CZ(wires=wires[::-1]))
    op_list.append(qml.CRot(0, 2 * theta, 0, wires=wires[::-1]))
    op_list.extend(decompose_ua(-phi, wires=wires))
    return op_list