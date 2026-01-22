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
@cache_readonly
def coefficients_of_determination(self):
    """
        Individual coefficients of determination (:math:`R^2`).

        Coefficients of determination (:math:`R^2`) from regressions of
        endogenous variables on individual estimated factors.

        Returns
        -------
        coefficients_of_determination : ndarray
            A `k_endog` x `k_factors` array, where
            `coefficients_of_determination[i, j]` represents the :math:`R^2`
            value from a regression of factor `j` and a constant on endogenous
            variable `i`.

        Notes
        -----
        Although it can be difficult to interpret the estimated factor loadings
        and factors, it is often helpful to use the coefficients of
        determination from univariate regressions to assess the importance of
        each factor in explaining the variation in each endogenous variable.

        In models with many variables and factors, this can sometimes lend
        interpretation to the factors (for example sometimes one factor will
        load primarily on real variables and another on nominal variables).

        See Also
        --------
        get_coefficients_of_determination
        plot_coefficients_of_determination
        """
    return self.get_coefficients_of_determination(method='individual')