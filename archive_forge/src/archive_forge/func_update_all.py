import pandas as pd
import numpy as np
import patsy
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.regression.linear_model import OLS
from collections import defaultdict
def update_all(self, n_iter=1):
    """
        Perform a specified number of MICE iterations.

        Parameters
        ----------
        n_iter : int
            The number of updates to perform.  Only the result of the
            final update will be available.

        Notes
        -----
        The imputed values are stored in the class attribute `self.data`.
        """
    for k in range(n_iter):
        for vname in self._cycle_order:
            self.update(vname)
    if self.history_callback is not None:
        hv = self.history_callback(self)
        self.history.append(hv)