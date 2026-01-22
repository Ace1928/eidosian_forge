from statsmodels.compat.pandas import MONTH_END, QUARTER_END
from collections import OrderedDict
from warnings import warn
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.validation import int_like
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.multivariate.pca import PCA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace._quarterly_ar1 import QuarterlyAR1
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tools.tools import Bunch
from statsmodels.tools.validation import string_like
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.statespace import mlemodel, initialization
from statsmodels.tsa.statespace.tools import (
from statsmodels.tsa.statespace.kalman_smoother import (
from statsmodels.base.data import PandasData
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.tableformatting import fmt_params
def get_coefficients_of_determination(self, method='individual', which=None):
    """
        Get coefficients of determination (R-squared) for variables / factors.

        Parameters
        ----------
        method : {'individual', 'joint', 'cumulative'}, optional
            The type of R-squared values to generate. "individual" plots
            the R-squared of each variable on each factor; "joint" plots the
            R-squared of each variable on each factor that it loads on;
            "cumulative" plots the successive R-squared values as each
            additional factor is added to the regression, for each variable.
            Default is 'individual'.
        which: {None, 'filtered', 'smoothed'}, optional
            Whether to compute R-squared values based on filtered or smoothed
            estimates of the factors. Default is 'smoothed' if smoothed results
            are available and 'filtered' otherwise.

        Returns
        -------
        rsquared : pd.DataFrame or pd.Series
            The R-squared values from regressions of observed variables on
            one or more of the factors. If method='individual' or
            method='cumulative', this will be a Pandas DataFrame with observed
            variables as the index and factors as the columns . If
            method='joint', will be a Pandas Series with observed variables as
            the index.

        See Also
        --------
        plot_coefficients_of_determination
        coefficients_of_determination
        """
    from statsmodels.tools import add_constant
    method = string_like(method, 'method', options=['individual', 'joint', 'cumulative'])
    if which is None:
        which = 'filtered' if self.smoothed_state is None else 'smoothed'
    k_endog = self.model.k_endog
    k_factors = self.model.k_factors
    ef_map = self.model._s.endog_factor_map
    endog_names = self.model.endog_names
    factor_names = self.model.factor_names
    if method == 'individual':
        coefficients = np.zeros((k_endog, k_factors))
        for i in range(k_factors):
            exog = add_constant(self.factors[which].iloc[:, i])
            for j in range(k_endog):
                if ef_map.iloc[j, i]:
                    endog = self.filter_results.endog[j]
                    coefficients[j, i] = OLS(endog, exog, missing='drop').fit().rsquared
                else:
                    coefficients[j, i] = np.nan
        coefficients = pd.DataFrame(coefficients, index=endog_names, columns=factor_names)
    elif method == 'joint':
        coefficients = np.zeros((k_endog,))
        exog = add_constant(self.factors[which])
        for j in range(k_endog):
            endog = self.filter_results.endog[j]
            ix = np.r_[True, ef_map.iloc[j]].tolist()
            X = exog.loc[:, ix]
            coefficients[j] = OLS(endog, X, missing='drop').fit().rsquared
        coefficients = pd.Series(coefficients, index=endog_names)
    elif method == 'cumulative':
        coefficients = np.zeros((k_endog, k_factors))
        exog = add_constant(self.factors[which])
        for j in range(k_endog):
            endog = self.filter_results.endog[j]
            for i in range(k_factors):
                if self.model._s.endog_factor_map.iloc[j, i]:
                    ix = np.r_[True, ef_map.iloc[j, :i + 1], [False] * (k_factors - i - 1)]
                    X = exog.loc[:, ix.astype(bool).tolist()]
                    coefficients[j, i] = OLS(endog, X, missing='drop').fit().rsquared
                else:
                    coefficients[j, i] = np.nan
        coefficients = pd.DataFrame(coefficients, index=endog_names, columns=factor_names)
    return coefficients