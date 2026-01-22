from warnings import simplefilter
import numpy as np
import wandb
from wandb.sklearn import calculate, utils
Logs a plot depicting model performance against dataset size.

    Please note this function fits the model to datasets of varying sizes when called.

    Arguments:
        model: (clf or reg) Takes in a fitted regressor or classifier.
        X: (arr) Dataset features.
        y: (arr) Dataset labels.

    For details on the other keyword arguments, see the documentation for
    `sklearn.model_selection.learning_curve`.

    Returns:
        None: To see plots, go to your W&B run page then expand the 'media' tab
              under 'auto visualizations'.

    Example:
    ```python
    wandb.sklearn.plot_learning_curve(model, X, y)
    ```
    