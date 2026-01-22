from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from pymatgen.core.tensors import Tensor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def get_piezo(BEC, IST, FCM, rcond=0.0001):
    """
    Generate a random piezoelectric tensor based on a structure and corresponding
    symmetry.

    Args:
        BEC (numpy array): Nx3x3 array representing the born effective charge tensor
        IST (numpy array): Nx3x3x3 array representing the internal strain tensor
        FCM (numpy array): NxNx3x3 array representing the born effective charge tensor
        rcondy (float): condition for excluding eigenvalues in the pseudoinverse

    Returns:
        3x3x3 calculated Piezo tensor
    """
    n_sites = len(BEC)
    temp_fcm = np.reshape(np.swapaxes(FCM, 1, 2), (n_sites * 3, n_sites * 3))
    eigs, _vecs = np.linalg.eig(temp_fcm)
    K = np.linalg.pinv(-temp_fcm, rcond=np.abs(eigs[np.argsort(np.abs(eigs))[2]]) / np.abs(eigs[np.argsort(np.abs(eigs))[-1]]) + rcond)
    K = np.reshape(K, (n_sites, 3, n_sites, 3)).swapaxes(1, 2)
    return np.einsum('ikl,ijlm,jmno->kno', BEC, K, IST) * 16.0216559424