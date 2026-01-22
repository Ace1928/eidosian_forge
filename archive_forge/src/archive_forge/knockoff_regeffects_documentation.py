import numpy as np

    Use any regression model for Regression FDR analysis.

    Parameters
    ----------
    parent : RegressionFDR
        The RegressionFDR instance to which this effect size is
        applied.
    model_cls : class
        Any model with appropriate fit or fit_regularized
        functions
    regularized : bool
        If True, use fit_regularized to fit the model
    model_kws : dict
        Keywords passed to model initializer
    fit_kws : dict
        Dictionary of keyword arguments for fit or fit_regularized
    