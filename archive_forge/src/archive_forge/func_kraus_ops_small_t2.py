import warnings
from pennylane import math as np
from pennylane.operation import AnyWires, Channel
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