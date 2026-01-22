from __future__ import annotations
from typing import TYPE_CHECKING, ClassVar
import numpy as np
from pymatgen.analysis.chemenv.utils.math_utils import (
def inverse_smoothstep(self, vals):
    """Get the evaluation of the "inverse" smoothstep ratio function: f(x)=1-(3*x^2-2*x^3).

        The values (i.e. "x"), are scaled between the "lower" and "upper" parameters.

        Args:
            vals: Values for which the ratio function has to be evaluated.

        Returns:
            Result of the ratio function applied to the values.
        """
    return smoothstep(vals, edges=[self.__dict__['lower'], self.__dict__['upper']], inverse=True)