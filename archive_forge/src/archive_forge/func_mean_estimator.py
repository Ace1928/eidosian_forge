from __future__ import annotations
from typing import TYPE_CHECKING, ClassVar
import numpy as np
from pymatgen.analysis.chemenv.utils.math_utils import (
def mean_estimator(self, data):
    """Get the weighted CSM using this CSM ratio function applied to the data.

        Args:
            data: List of CSM values to estimate the weighted CSM.

        Returns:
            Weighted CSM from this ratio function.
        """
    if len(data) == 0:
        return None
    if len(data) == 1:
        return data[0]
    fractions = self.fractions(data)
    if fractions is None:
        return None
    return np.sum(np.array(fractions) * np.array(data))