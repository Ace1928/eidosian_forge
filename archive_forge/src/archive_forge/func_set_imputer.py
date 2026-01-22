import pandas as pd
import numpy as np
import patsy
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.regression.linear_model import OLS
from collections import defaultdict
def set_imputer(self, endog_name, formula=None, model_class=None, init_kwds=None, fit_kwds=None, predict_kwds=None, k_pmm=20, perturbation_method=None, regularized=False):
    """
        Specify the imputation process for a single variable.

        Parameters
        ----------
        endog_name : str
            Name of the variable to be imputed.
        formula : str
            Conditional formula for imputation. Defaults to a formula
            with main effects for all other variables in dataset.  The
            formula should only include an expression for the mean
            structure, e.g. use 'x1 + x2' not 'x4 ~ x1 + x2'.
        model_class : statsmodels model
            Conditional model for imputation. Defaults to OLS.  See below
            for more information.
        init_kwds : dit-like
            Keyword arguments passed to the model init method.
        fit_kwds : dict-like
            Keyword arguments passed to the model fit method.
        predict_kwds : dict-like
            Keyword arguments passed to the model predict method.
        k_pmm : int
            Determines number of neighboring observations from which
            to randomly sample when using predictive mean matching.
        perturbation_method : str
            Either 'gaussian' or 'bootstrap'. Determines the method
            for perturbing parameters in the imputation model.  If
            None, uses the default specified at class initialization.
        regularized : dict
            If regularized[name]=True, `fit_regularized` rather than
            `fit` is called when fitting imputation models for this
            variable.  When regularized[name]=True for any variable,
            perturbation_method must be set to boot.

        Notes
        -----
        The model class must meet the following conditions:
            * A model must have a 'fit' method that returns an object.
            * The object returned from `fit` must have a `params` attribute
              that is an array-like object.
            * The object returned from `fit` must have a cov_params method
              that returns a square array-like object.
            * The model must have a `predict` method.
        """
    if formula is None:
        main_effects = [x for x in self.data.columns if x != endog_name]
        fml = endog_name + ' ~ ' + ' + '.join(main_effects)
        self.conditional_formula[endog_name] = fml
    else:
        fml = endog_name + ' ~ ' + formula
        self.conditional_formula[endog_name] = fml
    if model_class is None:
        self.model_class[endog_name] = OLS
    else:
        self.model_class[endog_name] = model_class
    if init_kwds is not None:
        self.init_kwds[endog_name] = init_kwds
    if fit_kwds is not None:
        self.fit_kwds[endog_name] = fit_kwds
    if predict_kwds is not None:
        self.predict_kwds[endog_name] = predict_kwds
    if perturbation_method is not None:
        self.perturbation_method[endog_name] = perturbation_method
    self.k_pmm = k_pmm
    self.regularized[endog_name] = regularized