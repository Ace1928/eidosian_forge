from warnings import simplefilter
import numpy as np
import wandb
from wandb.sklearn import calculate, utils
from . import shared
Measures and plots the regressor's predicted value against the residual.

    The marginal distribution of residuals is also calculated and plotted.

    Should only be called with a fitted regressor (otherwise an error is thrown).

    Please note this function fits variations of the model on the training set when called.

    Arguments:
        regressor: (regressor) Takes in a fitted regressor.
        X: (arr) Training set features.
        y: (arr) Training set labels.

    Returns:
        None: To see plots, go to your W&B run page then expand the 'media' tab
              under 'auto visualizations'.

    Example:
    ```python
    wandb.sklearn.plot_residuals(model, X, y)
    ```
    