import warnings
from pennylane import math as np
from pennylane.operation import AnyWires, Channel
@staticmethod
def compute_kraus_matrices(pe, t1, t2, tg):
    """Kraus matrices representing the ThermalRelaxationError channel.

        Args:
            pe (float): exited state population. Must be between ``0`` and ``1``
            t1 (float): the :math:`T_1` relaxation constant
            t2 (float): The :math:`T_2` dephasing constant. Must be less than :math:`2 T_1`
            tg (float): the gate time for relaxation error

        Returns:
            list (array): list of Kraus matrices

        **Example**

        >>> qml.ThermalRelaxationError.compute_kraus_matrices(0.1, 1.2, 1.3, 0.1)
        [array([[0.        , 0.        ], [0.08941789, 0.        ]]),
         array([[0.        , 0.26825366], [0.        , 0.        ]]),
         array([[-0.12718544,  0.        ], [ 0.        ,  0.13165421]]),
         array([[0.98784022, 0.        ], [0.        , 0.95430977]])]
        """
    if not np.is_abstract(pe) and (not 0.0 <= pe <= 1.0):
        raise ValueError('pe must be between 0 and 1.')
    if not np.is_abstract(tg) and tg < 0:
        raise ValueError(f'Invalid gate_time tg ({tg} < 0)')
    if not np.is_abstract(t1) and t1 <= 0:
        raise ValueError('Invalid T_1 relaxation time parameter: T_1 <= 0.')
    if not np.is_abstract(t2) and t2 <= 0:
        raise ValueError('Invalid T_2 relaxation time parameter: T_2 <= 0.')
    if not np.is_abstract(t2 - 2 * t1) and t2 - 2 * t1 > 0:
        raise ValueError('Invalid T_2 relaxation time parameter: T_2 greater than 2 * T_1.')
    eT1 = np.exp(-tg / t1)
    p_reset = 1 - eT1
    eT2 = np.exp(-tg / t2)

    def kraus_ops_small_t2():
        pz = (1 - p_reset) * (1 - eT2 / eT1) / 2
        pr0 = (1 - pe) * p_reset
        pr1 = pe * p_reset
        pid = 1 - pz - pr0 - pr1
        K0 = np.sqrt(pid + np.eps) * np.eye(2)
        K1 = np.sqrt(pz + np.eps) * np.array([[1, 0], [0, -1]])
        K2 = np.sqrt(pr0 + np.eps) * np.array([[1, 0], [0, 0]])
        K3 = np.sqrt(pr0 + np.eps) * np.array([[0, 1], [0, 0]])
        K4 = np.sqrt(pr1 + np.eps) * np.array([[0, 0], [1, 0]])
        K5 = np.sqrt(pr1 + np.eps) * np.array([[0, 0], [0, 1]])
        return [K0, K1, K2, K3, K4, K5]

    def kraus_ops_large_t2():
        e0 = p_reset * pe
        v0 = np.array([[0, 0], [1, 0]])
        K0 = np.sqrt(e0 + np.eps) * v0
        e1 = -p_reset * pe + p_reset
        v1 = np.array([[0, 1], [0, 0]])
        K1 = np.sqrt(e1 + np.eps) * v1
        base = sum((4 * eT2 ** 2, 4 * p_reset ** 2 * pe ** 2, -4 * p_reset ** 2 * pe, p_reset ** 2, np.eps))
        common_term = np.sqrt(base)
        e2 = 1 - p_reset / 2 - common_term / 2
        term2 = 2 * eT2 / (2 * p_reset * pe - p_reset - common_term)
        v2 = (term2 * np.array([[1, 0], [0, 0]]) + np.array([[0, 0], [0, 1]])) / np.sqrt(term2 ** 2 + 1)
        K2 = np.sqrt(e2 + np.eps) * v2
        term3 = 2 * eT2 / (2 * p_reset * pe - p_reset + common_term)
        e3 = 1 - p_reset / 2 + common_term / 2
        v3 = (term3 * np.array([[1, 0], [0, 0]]) + np.array([[0, 0], [0, 1]])) / np.sqrt(term3 ** 2 + 1)
        K3 = np.sqrt(e3 + np.eps) * v3
        K4 = np.cast_like(np.zeros((2, 2)), K1)
        K5 = np.cast_like(np.zeros((2, 2)), K1)
        return [K0, K1, K2, K3, K4, K5]
    K = np.cond(t2 <= t1, kraus_ops_small_t2, kraus_ops_large_t2, ())
    return K