from __future__ import annotations
import logging
import numpy as np
from qiskit.exceptions import QiskitError, MissingOptionalLibraryError
from qiskit.circuit.gate import Gate
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.channel import Choi, SuperOp
from qiskit.quantum_info.states.densitymatrix import DensityMatrix
from qiskit.quantum_info.states.measures import state_fidelity
from qiskit.utils import optionals as _optionals
def process_fidelity(channel: Operator | QuantumChannel, target: Operator | QuantumChannel | None=None, require_cp: bool=True, require_tp: bool=True) -> float:
    """Return the process fidelity of a noisy quantum channel.


    The process fidelity :math:`F_{\\text{pro}}(\\mathcal{E}, \\mathcal{F})`
    between two quantum channels :math:`\\mathcal{E}, \\mathcal{F}` is given by

    .. math::
        F_{\\text{pro}}(\\mathcal{E}, \\mathcal{F})
            = F(\\rho_{\\mathcal{E}}, \\rho_{\\mathcal{F}})

    where :math:`F` is the :func:`~qiskit.quantum_info.state_fidelity`,
    :math:`\\rho_{\\mathcal{E}} = \\Lambda_{\\mathcal{E}} / d` is the
    normalized :class:`~qiskit.quantum_info.Choi` matrix for the channel
    :math:`\\mathcal{E}`, and :math:`d` is the input dimension of
    :math:`\\mathcal{E}`.

    When the target channel is unitary this is equivalent to

    .. math::
        F_{\\text{pro}}(\\mathcal{E}, U)
            = \\frac{Tr[S_U^\\dagger S_{\\mathcal{E}}]}{d^2}

    where :math:`S_{\\mathcal{E}}, S_{U}` are the
    :class:`~qiskit.quantum_info.SuperOp` matrices for the *input* quantum
    channel :math:`\\mathcal{E}` and *target* unitary :math:`U` respectively,
    and :math:`d` is the input dimension of the channel.

    Args:
        channel (Operator or QuantumChannel): input quantum channel.
        target (Operator or QuantumChannel or None): target quantum channel.
            If `None` target is the identity operator [Default: None].
        require_cp (bool): check if input and target channels are
                           completely-positive and if non-CP log warning
                           containing negative eigenvalues of Choi-matrix
                           [Default: True].
        require_tp (bool): check if input and target channels are
                           trace-preserving and if non-TP log warning
                           containing negative eigenvalues of partial
                           Choi-matrix :math:`Tr_{\\text{out}}[\\mathcal{E}] - I`
                           [Default: True].

    Returns:
        float: The process fidelity :math:`F_{\\text{pro}}`.

    Raises:
        QiskitError: if the channel and target do not have the same dimensions.
    """
    channel = _input_formatter(channel, SuperOp, 'process_fidelity', 'channel')
    target = _input_formatter(target, Operator, 'process_fidelity', 'target')
    if target:
        if channel.dim != target.dim:
            raise QiskitError('Input quantum channel and target unitary must have the same dimensions ({} != {}).'.format(channel.dim, target.dim))
    for label, chan in [('Input', channel), ('Target', target)]:
        if chan is not None and require_cp:
            cp_cond = _cp_condition(chan)
            neg = cp_cond < -1 * chan.atol
            if np.any(neg):
                logger.warning('%s channel is not CP. Choi-matrix has negative eigenvalues: %s', label, cp_cond[neg])
        if chan is not None and require_tp:
            tp_cond = _tp_condition(chan)
            non_zero = np.logical_not(np.isclose(tp_cond, 0, atol=chan.atol, rtol=chan.rtol))
            if np.any(non_zero):
                logger.warning('%s channel is not TP. Tr_2[Choi] - I has non-zero eigenvalues: %s', label, tp_cond[non_zero])
    if isinstance(target, Operator):
        channel = channel.compose(target.adjoint())
        target = None
    input_dim, _ = channel.dim
    if target is None:
        if isinstance(channel, Operator):
            fid = np.abs(np.trace(channel.data) / input_dim) ** 2
        else:
            fid = np.trace(SuperOp(channel).data) / input_dim ** 2
        return float(np.real(fid))
    state1 = DensityMatrix(Choi(channel).data / input_dim)
    state2 = DensityMatrix(Choi(target).data / input_dim)
    return state_fidelity(state1, state2, validate=False)