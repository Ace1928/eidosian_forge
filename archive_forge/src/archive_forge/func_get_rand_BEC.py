from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from pymatgen.core.tensors import Tensor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def get_rand_BEC(self, max_charge=1):
    """
        Generate a random born effective charge tensor which obeys a structure's
        symmetry and the acoustic sum rule.

        Args:
            max_charge (float): maximum born effective charge value

        Returns:
            np.array Born effective charge tensor
        """
    n_atoms = len(self.structure)
    BEC = np.zeros((n_atoms, 3, 3))
    for atom, ops in enumerate(self.BEC_operations):
        if ops[0] == ops[1]:
            temp_tensor = Tensor(np.random.rand(3, 3) - 0.5)
            temp_tensor = sum((temp_tensor.transform(symm_op) for symm_op in self.pointops[atom])) / len(self.pointops[atom])
            BEC[atom] = temp_tensor
        else:
            temp_fcm = np.zeros([3, 3])
            for op in ops[2]:
                temp_fcm += op.transform_tensor(BEC[self.BEC_operations[atom][1]])
            BEC[ops[0]] = temp_fcm
            if len(ops[2]) != 0:
                BEC[ops[0]] = BEC[ops[0]] / len(ops[2])
    disp_charge = np.einsum('ijk->jk', BEC) / n_atoms
    add = np.zeros([n_atoms, 3, 3])
    for atom, ops in enumerate(self.BEC_operations):
        if ops[0] == ops[1]:
            temp_tensor = Tensor(disp_charge)
            temp_tensor = sum((temp_tensor.transform(symm_op) for symm_op in self.pointops[atom])) / len(self.pointops[atom])
            add[ops[0]] = temp_tensor
        else:
            temp_tensor = np.zeros([3, 3])
            for op in ops[2]:
                temp_tensor += op.transform_tensor(add[self.BEC_operations[atom][1]])
            add[ops[0]] = temp_tensor
            if len(ops) != 0:
                add[ops[0]] = add[ops[0]] / len(ops[2])
    BEC = BEC - add
    return BEC * max_charge