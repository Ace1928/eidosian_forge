from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from pymatgen.core.tensors import Tensor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
@requires(Phonopy, 'phonopy not installed!')
def rand_piezo(struct, pointops, sharedops, BEC, IST, FCM, anumiter=10):
    """
    Generate a random piezoelectric tensor based on a structure and corresponding
    symmetry.

    Args:
        struct (pymatgen structure): structure whose symmetry operations the piezo tensor must obey
        pointops: list of point operations obeyed by a single atomic site
        sharedops: list of point operations shared by a pair of atomic sites
        BEC (numpy array): Nx3x3 array representing the born effective charge tensor
        IST (numpy array): Nx3x3x3 array representing the internal strain tensor
        FCM (numpy array): NxNx3x3 array representing the born effective charge tensor
        anumiter (int): number of iterations for acoustic sum rule convergence
    Returns:
        list in the form of [Nx3x3 random born effective charge tenosr,
        Nx3x3x3 random internal strain tensor, NxNx3x3 random force constant matrix, 3x3x3 piezo tensor]
    """
    bec = BornEffectiveCharge(struct, BEC, pointops)
    bec.get_BEC_operations()
    rand_BEC = bec.get_rand_BEC()
    ist = InternalStrainTensor(struct, IST, pointops)
    ist.get_IST_operations()
    rand_IST = ist.get_rand_IST()
    fcm = ForceConstantMatrix(struct, FCM, pointops, sharedops)
    fcm.get_FCM_operations()
    rand_FCM = fcm.get_rand_FCM()
    P = get_piezo(rand_BEC, rand_IST, rand_FCM) * 16.0216559424 / struct.volume
    return (rand_BEC, rand_IST, rand_FCM, P)