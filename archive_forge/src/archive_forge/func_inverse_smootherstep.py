from __future__ import annotations
from typing import TYPE_CHECKING, ClassVar
import numpy as np
from pymatgen.analysis.chemenv.utils.math_utils import (
def inverse_smootherstep(self, vals):
    """Get the evaluation of the "inverse" smootherstep ratio function: f(x)=1-(6*x^5-15*x^4+10*x^3).

        The values (i.e. "x"), are scaled between the "lower" and "upper" parameters.

        Args:
            vals: Values for which the ratio function has to be evaluated.

        Returns:
            Result of the ratio function applied to the values.
        """
    return smootherstep(vals, edges=[self.__dict__['lower'], self.__dict__['upper']], inverse=True)