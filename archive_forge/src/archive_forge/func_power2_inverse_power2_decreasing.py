from __future__ import annotations
from typing import TYPE_CHECKING, ClassVar
import numpy as np
from pymatgen.analysis.chemenv.utils.math_utils import (
def power2_inverse_power2_decreasing(self, vals):
    """Get the evaluation of the ratio function f(x)=(x-1)^2 / x^2.

        The CSM values (i.e. "x"), are scaled to the "max_csm" parameter. The "a" constant
        correspond to the "alpha" parameter.

        Args:
            vals: CSM values for which the ratio function has to be evaluated.

        Returns:
            Result of the ratio function applied to the CSM values.
        """
    return power2_inverse_power2_decreasing(vals, edges=[0.0, self.__dict__['max_csm']])