import numpy as np
import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.ops import QubitUnitary
def make_Q(A, R):
    """Calculates the :math:`\\mathcal{Q}` matrix that encodes the expectation value according to
    the probability unitary :math:`\\mathcal{A}` and the function-encoding unitary
    :math:`\\mathcal{R}`.

    Following `this <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.98.022321>`__ paper,
    the expectation value is encoded as the phase of an eigenvalue of :math:`\\mathcal{Q}`. This
    phase can be estimated using quantum phase estimation using the
    :func:`~.QuantumPhaseEstimation` template. See :func:`~.QuantumMonteCarlo` for more details,
    which loads ``make_Q()`` internally and applies phase estimation.

    Args:
        A (array): The unitary matrix of :math:`\\mathcal{A}` which encodes the probability
            distribution
        R (array): The unitary matrix of :math:`\\mathcal{R}` which encodes the function

    Returns:
        array: the :math:`\\mathcal{Q}` unitary
    """
    A_big = qml.math.kron(A, np.eye(2))
    F = R @ A_big
    F_dagger = F.conj().T
    dim = len(R)
    V = _make_V(dim)
    Z = _make_Z(dim)
    UV = F @ Z @ F_dagger @ V
    return UV @ UV