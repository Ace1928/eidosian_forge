import numpy as np
from statsmodels.compat.pandas import Substitution
from statsmodels.base.model import Model
from .multivariate_ols import MultivariateTestResults
from .multivariate_ols import _multivariate_ols_fit
from .multivariate_ols import _multivariate_ols_test, _hypotheses_doc

        Linear hypotheses testing

        Parameters
        ----------
        %(hypotheses_doc)s
        skip_intercept_test : bool
            If true, then testing the intercept is skipped, the model is not
            changed.
            Note: If a term has a numerically insignificant effect, then
            an exception because of emtpy arrays may be raised. This can
            happen for the intercept if the data has been demeaned.

        Returns
        -------
        results: MultivariateTestResults

        Notes
        -----
        Testing the linear hypotheses

            L * params * M = 0

        where `params` is the regression coefficient matrix for the
        linear model y = x * params

        If the model is not specified using the formula interfact, then the
        hypotheses test each included exogenous variable, one at a time. In
        most applications with categorical variables, the ``from_formula``
        interface should be preferred when specifying a model since it
        provides knowledge about the model when specifying the hypotheses.
        