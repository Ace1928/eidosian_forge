import numpy as np
import statsmodels.base.wrapper as wrap
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.regime_switching import (
from statsmodels.tsa.statespace.tools import (
def predict_conditional(self, params):
    """
        In-sample prediction, conditional on the current and previous regime

        Parameters
        ----------
        params : array_like
            Array of parameters at which to create predictions.

        Returns
        -------
        predict : array_like
            Array of predictions conditional on current, and possibly past,
            regimes
        """
    params = np.array(params, ndmin=1)
    if self._k_exog > 0:
        xb = []
        for i in range(self.k_regimes):
            coeffs = params[self.parameters[i, 'exog']]
            xb.append(np.dot(self.orig_exog, coeffs))
    predict = np.zeros((self.k_regimes,) * (self.order + 1) + (self.nobs,), dtype=np.promote_types(np.float64, params.dtype))
    for i in range(self.k_regimes):
        ar_coeffs = params[self.parameters[i, 'autoregressive']]
        ix = self._predict_slices[:]
        ix[0] = i
        ix = tuple(ix)
        if self._k_exog > 0:
            predict[ix] += xb[i][self.order:]
        for j in range(1, self.order + 1):
            for k in range(self.k_regimes):
                ix = self._predict_slices[:]
                ix[0] = i
                ix[j] = k
                ix = tuple(ix)
                start = self.order - j
                end = -j
                if self._k_exog > 0:
                    predict[ix] += ar_coeffs[j - 1] * (self.orig_endog[start:end] - xb[k][start:end])
                else:
                    predict[ix] += ar_coeffs[j - 1] * self.orig_endog[start:end]
    return predict