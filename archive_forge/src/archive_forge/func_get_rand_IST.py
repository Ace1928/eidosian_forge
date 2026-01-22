from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from pymatgen.core.tensors import Tensor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def get_rand_IST(self, max_force=1):
    """
        Generate a random internal strain tensor which obeys a structure's
        symmetry and the acoustic sum rule.

        Args:
            max_force (float): maximum born effective charge value

        Returns:
            InternalStrainTensor
        """
    n_atoms = len(self.structure)
    IST = np.zeros((n_atoms, 3, 3, 3))
    for atom, ops in enumerate(self.IST_operations):
        temp_tensor = np.zeros([3, 3, 3])
        for op in ops:
            temp_tensor += op[1].transform_tensor(IST[op[0]])
        if len(ops) == 0:
            temp_tensor = Tensor(np.random.rand(3, 3, 3) - 0.5)
            for dim in range(3):
                temp_tensor[dim] = (temp_tensor[dim] + temp_tensor[dim].T) / 2
            temp_tensor = sum((temp_tensor.transform(symm_op) for symm_op in self.pointops[atom])) / len(self.pointops[atom])
        IST[atom] = temp_tensor
        if len(ops) != 0:
            IST[atom] = IST[atom] / len(ops)
    return IST * max_force