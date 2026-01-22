import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires
def decompose_ua(phi, wires=None):
    """Appends the circuit decomposing the controlled application of the unitary
    :math:`U_A(\\phi)`

    .. math::

        U_A(\\phi) = \\left(\\begin{array}{cc} 0 & e^{-i\\phi} \\\\ e^{-i\\phi} & 0 \\\\ \\end{array}\\right)

    in terms of the quantum operations supported by PennyLane.

    :math:`U_A(\\phi)` is used in `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_,
    to define two-qubit exchange gates required to build particle-conserving
    VQE ansatze for quantum chemistry simulations. See :func:`~.ParticleConservingU1`.

    :math:`U_A(\\phi)` is expressed in terms of ``PhaseShift``, ``Rot`` and ``PauliZ`` operations
    :math:`U_A(\\phi) = R_\\phi(-2\\phi) R(-\\phi, \\pi, \\phi) \\sigma_z`.

    Args:
        phi (float): angle :math:`\\phi` defining the unitary :math:`U_A(\\phi)`
        wires (Iterable): the wires ``n`` and ``m`` the circuit acts on

    Returns:
          list[.Operator]: sequence of operators defined by this function
    """
    op_list = []
    n, m = wires
    op_list.append(qml.CZ(wires=wires))
    op_list.append(qml.CRot(-phi, np.pi, phi, wires=wires))
    op_list.append(qml.PhaseShift(-phi, wires=m))
    op_list.append(qml.CNOT(wires=wires))
    op_list.append(qml.PhaseShift(phi, wires=m))
    op_list.append(qml.CNOT(wires=wires))
    op_list.append(qml.PhaseShift(-phi, wires=n))
    return op_list