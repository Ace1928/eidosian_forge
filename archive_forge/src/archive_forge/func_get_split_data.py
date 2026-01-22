import pandas as pd
import numpy as np
import patsy
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.regression.linear_model import OLS
from collections import defaultdict
def get_split_data(self, vname):
    """
        Return endog and exog for imputation of a given variable.

        Parameters
        ----------
        vname : str
           The variable for which the split data is returned.

        Returns
        -------
        endog_obs : DataFrame
            Observed values of the variable to be imputed.
        exog_obs : DataFrame
            Current values of the predictors where the variable to be
            imputed is observed.
        exog_miss : DataFrame
            Current values of the predictors where the variable to be
            Imputed is missing.
        init_kwds : dict-like
            The init keyword arguments for `vname`, processed through Patsy
            as required.
        fit_kwds : dict-like
            The fit keyword arguments for `vname`, processed through Patsy
            as required.
        """
    formula = self.conditional_formula[vname]
    endog, exog = patsy.dmatrices(formula, self.data, return_type='dataframe')
    ixo = self.ix_obs[vname]
    endog_obs = np.require(endog.iloc[ixo], requirements='W')
    exog_obs = np.require(exog.iloc[ixo, :], requirements='W')
    ixm = self.ix_miss[vname]
    exog_miss = np.require(exog.iloc[ixm, :], requirements='W')
    predict_obs_kwds = {}
    if vname in self.predict_kwds:
        kwds = self.predict_kwds[vname]
        predict_obs_kwds = self._process_kwds(kwds, ixo)
    predict_miss_kwds = {}
    if vname in self.predict_kwds:
        kwds = self.predict_kwds[vname]
        predict_miss_kwds = self._process_kwds(kwds, ixo)
    return (endog_obs, exog_obs, exog_miss, predict_obs_kwds, predict_miss_kwds)