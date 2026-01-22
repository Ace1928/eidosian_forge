from warnings import simplefilter
import numpy as np
import wandb
from wandb.sklearn import calculate, utils
from . import shared
def regressor(model, X_train, X_test, y_train, y_test, model_name='Regressor'):
    """Generates all sklearn regressor plots supported by W&B.

    The following plots are generated:
        learning curve, summary metrics, residuals plot, outlier candidates.

    Should only be called with a fitted regressor (otherwise an error is thrown).

    Arguments:
        model: (regressor) Takes in a fitted regressor.
        X_train: (arr) Training set features.
        y_train: (arr) Training set labels.
        X_test: (arr) Test set features.
        y_test: (arr) Test set labels.
        model_name: (str) Model name. Defaults to 'Regressor'

    Returns:
        None: To see plots, go to your W&B run page then expand the 'media' tab
            under 'auto visualizations'.

    Example:
    ```python
    wandb.sklearn.plot_regressor(reg, X_train, X_test, y_train, y_test, "Ridge")
    ```
    """
    wandb.termlog('\nPlotting %s.' % model_name)
    shared.summary_metrics(model, X_train, y_train, X_test, y_test)
    wandb.termlog('Logged summary metrics.')
    shared.learning_curve(model, X_train, y_train)
    wandb.termlog('Logged learning curve.')
    outlier_candidates(model, X_train, y_train)
    wandb.termlog('Logged outlier candidates.')
    residuals(model, X_train, y_train)
    wandb.termlog('Logged residuals.')