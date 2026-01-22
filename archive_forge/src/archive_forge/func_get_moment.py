from __future__ import annotations
from enum import Enum, unique
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
def get_moment(self, saxis=(0, 0, 1)):
    """Get magnetic moment relative to a given spin quantization axis.
        If no axis is provided, moment will be given relative to the
        Magmom's internal spin quantization axis, i.e. equivalent to
        Magmom.moment.

        Args:
            saxis: (list/numpy array) spin quantization axis

        Returns:
            np.ndarray of length 3
        """
    m_inv = self._get_transformation_matrix_inv(self.saxis)
    moment = np.matmul(self.moment, m_inv)
    m = self._get_transformation_matrix(saxis)
    moment = np.matmul(moment, m)
    moment[np.abs(moment) < 1e-08] = 0
    return moment