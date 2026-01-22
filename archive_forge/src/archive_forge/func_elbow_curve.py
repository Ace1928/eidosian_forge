from warnings import simplefilter
import pandas as pd
import sklearn
import wandb
from wandb.sklearn import calculate, utils
def elbow_curve(clusterer=None, X=None, cluster_ranges=None, n_jobs=1, show_cluster_time=True):
    """Measures and plots variance explained as a function of the number of clusters.

    Useful in picking the optimal number of clusters.

    Should only be called with a fitted clusterer (otherwise an error is thrown).

    Please note this function fits the model on the training set when called.

    Arguments:
        model: (clusterer) Takes in a fitted clusterer.
        X: (arr) Training set features.

    Returns:
        None: To see plots, go to your W&B run page then expand the 'media' tab
              under 'auto visualizations'.

    Example:
    ```python
    wandb.sklearn.plot_elbow_curve(model, X_train)
    ```
    """
    if not hasattr(clusterer, 'n_clusters'):
        wandb.termlog('n_clusters attribute not in classifier. Cannot plot elbow method.')
        return
    not_missing = utils.test_missing(clusterer=clusterer)
    correct_types = utils.test_types
    is_fitted = utils.test_fitted(clusterer)
    if not_missing and correct_types and is_fitted:
        elbow_curve_chart = calculate.elbow_curve(clusterer, X, cluster_ranges, n_jobs, show_cluster_time)
        wandb.log({'elbow_curve': elbow_curve_chart})