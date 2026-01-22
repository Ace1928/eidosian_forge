from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from pymatgen.core.tensors import Tensor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def get_asum_FCM(self, fcm: np.ndarray, numiter: int=15):
    """
        Generate a symmetrized force constant matrix that obeys the objects symmetry
        constraints and obeys the acoustic sum rule through an iterative procedure.

        Args:
            fcm (numpy array): 3Nx3N unsymmetrized force constant matrix
            numiter (int): number of iterations to attempt to obey the acoustic sum
                rule

        Returns:
            numpy array representing the force constant matrix
        """
    operations = self.FCM_operations
    if operations is None:
        raise RuntimeError('No symmetry operations found. Run get_FCM_operations first.')
    n_sites = len(self.structure)
    D = np.ones([n_sites * 3, n_sites * 3])
    for _ in range(numiter):
        X = np.real(fcm)
        pastrow = 0
        total = np.zeros([3, 3])
        for col in range(n_sites):
            total = total + X[0:3, col * 3:col * 3 + 3]
        total = total / n_sites
        for op in operations:
            same = 0
            transpose = 0
            if op[0] == op[1] and op[0] == op[2] and (op[0] == op[3]):
                same = 1
            if op[0] == op[3] and op[1] == op[2]:
                transpose = 1
            if transpose == 0 and same == 0:
                D[3 * op[0]:3 * op[0] + 3, 3 * op[1]:3 * op[1] + 3] = np.zeros([3, 3])
                for symop in op[4]:
                    tempfcm = D[3 * op[2]:3 * op[2] + 3, 3 * op[3]:3 * op[3] + 3]
                    tempfcm = symop.transform_tensor(tempfcm)
                    D[3 * op[0]:3 * op[0] + 3, 3 * op[1]:3 * op[1] + 3] += tempfcm
                if len(op[4]) != 0:
                    D[3 * op[0]:3 * op[0] + 3, 3 * op[1]:3 * op[1] + 3] = D[3 * op[0]:3 * op[0] + 3, 3 * op[1]:3 * op[1] + 3] / len(op[4])
                D[3 * op[1]:3 * op[1] + 3, 3 * op[0]:3 * op[0] + 3] = D[3 * op[0]:3 * op[0] + 3, 3 * op[1]:3 * op[1] + 3].T
                continue
            curr_row = op[0]
            if curr_row != pastrow:
                total = np.zeros([3, 3])
                for col in range(n_sites):
                    total = total + X[curr_row * 3:curr_row * 3 + 3, col * 3:col * 3 + 3]
                for col in range(curr_row):
                    total = total - D[curr_row * 3:curr_row * 3 + 3, col * 3:col * 3 + 3]
                total = total / (n_sites - curr_row)
            pastrow = curr_row
            temp_tensor = Tensor(total)
            temp_tensor_sum = sum((temp_tensor.transform(symm_op) for symm_op in self.sharedops[op[0]][op[1]]))
            if len(self.sharedops[op[0]][op[1]]) != 0:
                temp_tensor_sum = temp_tensor_sum / len(self.sharedops[op[0]][op[1]])
            if op[0] != op[1]:
                for pair in range(len(op[4])):
                    temp_tensor2 = temp_tensor_sum.T
                    temp_tensor2 = op[4][pair].transform_tensor(temp_tensor2)
                    temp_tensor_sum = (temp_tensor_sum + temp_tensor2) / 2
            else:
                temp_tensor_sum = (temp_tensor_sum + temp_tensor_sum.T) / 2
            D[3 * op[0]:3 * op[0] + 3, 3 * op[1]:3 * op[1] + 3] = temp_tensor_sum
            D[3 * op[1]:3 * op[1] + 3, 3 * op[0]:3 * op[0] + 3] = temp_tensor_sum.T
        fcm = fcm - D
    return fcm