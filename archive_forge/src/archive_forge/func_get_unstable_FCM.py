from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from pymatgen.core.tensors import Tensor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def get_unstable_FCM(self, max_force=1):
    """
        Generate an unsymmetrized force constant matrix.

        Args:
            max_charge (float): maximum born effective charge value

        Returns:
            numpy array representing the force constant matrix
        """
    struct = self.structure
    operations = self.FCM_operations
    n_sites = len(struct)
    D = 1 / max_force * 2 * np.ones([n_sites * 3, n_sites * 3])
    for op in operations:
        same = 0
        transpose = 0
        if op[0] == op[1] and op[0] == op[2] and (op[0] == op[3]):
            same = 1
        if op[0] == op[3] and op[1] == op[2]:
            transpose = 1
        if transpose == 0 and same == 0:
            D[3 * op[0]:3 * op[0] + 3, 3 * op[1]:3 * op[1] + 3] = np.zeros([3, 3])
            D[3 * op[1]:3 * op[1] + 3, 3 * op[0]:3 * op[0] + 3] = np.zeros([3, 3])
            for symop in op[4]:
                temp_fcm = D[3 * op[2]:3 * op[2] + 3, 3 * op[3]:3 * op[3] + 3]
                temp_fcm = symop.transform_tensor(temp_fcm)
                D[3 * op[0]:3 * op[0] + 3, 3 * op[1]:3 * op[1] + 3] += temp_fcm
            if len(op[4]) != 0:
                D[3 * op[0]:3 * op[0] + 3, 3 * op[1]:3 * op[1] + 3] = D[3 * op[0]:3 * op[0] + 3, 3 * op[1]:3 * op[1] + 3] / len(op[4])
            D[3 * op[1]:3 * op[1] + 3, 3 * op[0]:3 * op[0] + 3] = D[3 * op[0]:3 * op[0] + 3, 3 * op[1]:3 * op[1] + 3].T
            continue
        temp_tensor = Tensor(np.random.rand(3, 3) - 0.5) * max_force
        temp_tensor_sum = sum((temp_tensor.transform(symm_op) for symm_op in self.sharedops[op[0]][op[1]]))
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
    return D