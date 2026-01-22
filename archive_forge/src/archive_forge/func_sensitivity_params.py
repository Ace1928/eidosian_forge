from statsmodels.compat.python import lzip
from statsmodels.compat.pandas import Appender
import numpy as np
from scipy import stats
import pandas as pd
import patsy
from collections import defaultdict
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.model as base
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM, GLMResults
from statsmodels.genmod import cov_struct as cov_structs
import statsmodels.genmod.families.varfuncs as varfuncs
from statsmodels.genmod.families.links import Link
from statsmodels.tools.sm_exceptions import (ConvergenceWarning,
import warnings
from statsmodels.graphics._regressionplots_doc import (
from statsmodels.discrete.discrete_margins import (
def sensitivity_params(self, dep_params_first, dep_params_last, num_steps):
    """
        Refits the GEE model using a sequence of values for the
        dependence parameters.

        Parameters
        ----------
        dep_params_first : array_like
            The first dep_params in the sequence
        dep_params_last : array_like
            The last dep_params in the sequence
        num_steps : int
            The number of dep_params in the sequence

        Returns
        -------
        results : array_like
            The GEEResults objects resulting from the fits.
        """
    model = self.model
    import copy
    cov_struct = copy.deepcopy(self.model.cov_struct)
    update_dep = model.update_dep
    model.update_dep = False
    dep_params = []
    results = []
    for x in np.linspace(0, 1, num_steps):
        dp = x * dep_params_last + (1 - x) * dep_params_first
        dep_params.append(dp)
        model.cov_struct = copy.deepcopy(cov_struct)
        model.cov_struct.dep_params = dp
        rslt = model.fit(start_params=self.params, ctol=self.ctol, params_niter=self.params_niter, first_dep_update=self.first_dep_update, cov_type=self.cov_type)
        results.append(rslt)
    model.update_dep = update_dep
    return results