from warnings import simplefilter
import numpy as np
from sklearn import naive_bayes
import wandb
import wandb.plots
from wandb.sklearn import calculate, utils
from . import shared
def class_proportions(y_train=None, y_test=None, labels=None):
    """Plot the distribution of target classses in training and test sets.

    Useful for detecting imbalanced classes.

    Arguments:
        y_train: (arr) Training set labels.
        y_test: (arr) Test set labels.
        labels: (list) Named labels for target variable (y). Makes plots easier to
                       read by replacing target values with corresponding index.
                       For example if `labels=['dog', 'cat', 'owl']` all 0s are
                       replaced by dog, 1s by cat.

    Returns:
        None: To see plots, go to your W&B run page then expand the 'media' tab
              under 'auto visualizations'.

    Example:
    ```python
    wandb.sklearn.plot_class_proportions(y_train, y_test, ["dog", "cat", "owl"])
    ```
    """
    not_missing = utils.test_missing(y_train=y_train, y_test=y_test)
    correct_types = utils.test_types(y_train=y_train, y_test=y_test)
    if not_missing and correct_types:
        y_train, y_test = (np.array(y_train), np.array(y_test))
        class_proportions_chart = calculate.class_proportions(y_train, y_test, labels)
        wandb.log({'class_proportions': class_proportions_chart})