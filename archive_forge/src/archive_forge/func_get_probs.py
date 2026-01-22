from statsmodels.compat.python import lrange
from pprint import pprint
import numpy as np
def get_probs(self, params):
    """
        obtain the probability array given an array of parameters

        This is the function that can be called by loglike or other methods
        that need the probabilities as function of the params.

        Parameters
        ----------
        params : 1d array, (nparams,)
            coefficients and tau that parameterize the model. The required
            length can be obtained by nparams. (and will depend on the number
            of degenerate leaves - not yet)

        Returns
        -------
        probs : ndarray, (nobs, nchoices)
            probabilities for all choices for each observation. The order
            is available by attribute leaves. See note in docstring of class



        """
    self.recursionparams = params
    self.calc_prob(self.tree)
    probs_array = np.array([self.probs[leaf] for leaf in self.leaves])
    return probs_array