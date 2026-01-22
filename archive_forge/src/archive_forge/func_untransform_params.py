import numpy as np
import statsmodels.base.wrapper as wrap
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.regime_switching import (
from statsmodels.tsa.statespace.tools import (
def untransform_params(self, constrained):
    """
        Transform constrained parameters used in likelihood evaluation
        to unconstrained parameters used by the optimizer

        Parameters
        ----------
        constrained : array_like
            Array of constrained parameters used in likelihood evaluation, to
            be transformed.

        Returns
        -------
        unconstrained : array_like
            Array of unconstrained parameters used by the optimizer.
        """
    unconstrained = super().untransform_params(constrained)
    for i in range(self.k_regimes):
        s = self.parameters[i, 'autoregressive']
        unconstrained[s] = unconstrain_stationary_univariate(constrained[s])
    return unconstrained