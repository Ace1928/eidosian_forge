from a common formula are constrained to have the same standard
import numpy as np
from scipy.optimize import minimize
from scipy import sparse
import statsmodels.base.model as base
from statsmodels.iolib import summary2
from statsmodels.genmod import families
import pandas as pd
import warnings
import patsy
def vb_elbo_grad(self, vb_mean, vb_sd):
    """
        Returns the gradient of the model's evidence lower bound (ELBO).
        """
    fep_mean, vcp_mean, vc_mean = self._unpack(vb_mean)
    fep_sd, vcp_sd, vc_sd = self._unpack(vb_sd)
    tm, tv = self._lp_stats(fep_mean, fep_sd, vc_mean, vc_sd)

    def h(z):
        y = -np.exp(tm + np.sqrt(tv) * z)
        return y
    return self.vb_elbo_grad_base(h, tm, tv, fep_mean, vcp_mean, vc_mean, fep_sd, vcp_sd, vc_sd)