from warnings import simplefilter
import pandas as pd
import sklearn
import wandb
from wandb.sklearn import calculate, utils
def silhouette(clusterer=None, X=None, cluster_labels=None, labels=None, metric='euclidean', kmeans=True):
    """Measures & plots silhouette coefficients.

    Silhouette coefficients near +1 indicate that the sample is far away from
    the neighboring clusters. A value near 0 indicates that the sample is on or
    very close to the decision boundary between two neighboring clusters and
    negative values indicate that the samples might have been assigned to the wrong cluster.

    Should only be called with a fitted clusterer (otherwise an error is thrown).

    Please note this function fits the model on the training set when called.

    Arguments:
        model: (clusterer) Takes in a fitted clusterer.
        X: (arr) Training set features.
        cluster_labels: (list) Names for cluster labels. Makes plots easier to read
                               by replacing cluster indexes with corresponding names.

    Returns:
        None: To see plots, go to your W&B run page then expand the 'media' tab
              under 'auto visualizations'.

    Example:
    ```python
    wandb.sklearn.plot_silhouette(model, X_train, ["spam", "not spam"])
    ```
    """
    not_missing = utils.test_missing(clusterer=clusterer)
    correct_types = utils.test_types(clusterer=clusterer)
    is_fitted = utils.test_fitted(clusterer)
    if not_missing and correct_types and is_fitted:
        if isinstance(X, pd.DataFrame):
            X = X.values
        silhouette_chart = calculate.silhouette(clusterer, X, cluster_labels, labels, metric, kmeans)
        wandb.log({'silhouette_plot': silhouette_chart})