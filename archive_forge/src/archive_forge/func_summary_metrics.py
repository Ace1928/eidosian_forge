from warnings import simplefilter
import numpy as np
import wandb
from wandb.sklearn import calculate, utils
def summary_metrics(model=None, X=None, y=None, X_test=None, y_test=None):
    """Logs a chart depicting summary metrics for a model.

    Should only be called with a fitted model (otherwise an error is thrown).

    Arguments:
        model: (clf or reg) Takes in a fitted regressor or classifier.
        X: (arr) Training set features.
        y: (arr) Training set labels.
        X_test: (arr) Test set features.
        y_test: (arr) Test set labels.

    Returns:
        None: To see plots, go to your W&B run page then expand the 'media' tab
              under 'auto visualizations'.

    Example:
    ```python
    wandb.sklearn.plot_summary_metrics(model, X_train, y_train, X_test, y_test)
    ```
    """
    not_missing = utils.test_missing(model=model, X=X, y=y, X_test=X_test, y_test=y_test)
    correct_types = utils.test_types(model=model, X=X, y=y, X_test=X_test, y_test=y_test)
    model_fitted = utils.test_fitted(model)
    if not_missing and correct_types and model_fitted:
        metrics_chart = calculate.summary_metrics(model, X, y, X_test, y_test)
        wandb.log({'summary_metrics': metrics_chart})