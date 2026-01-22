from warnings import simplefilter
import pandas as pd
import sklearn
import wandb
from wandb.sklearn import calculate, utils
def clusterer(model, X_train, cluster_labels, labels=None, model_name='Clusterer'):
    """Generates all sklearn clusterer plots supported by W&B.

    The following plots are generated:
        elbow curve, silhouette plot.

    Should only be called with a fitted clusterer (otherwise an error is thrown).

    Arguments:
        model: (clusterer) Takes in a fitted clusterer.
        X_train: (arr) Training set features.
        cluster_labels: (list) Names for cluster labels. Makes plots easier to read
                            by replacing cluster indexes with corresponding names.
        labels: (list) Named labels for target varible (y). Makes plots easier to
                        read by replacing target values with corresponding index.
                        For example if `labels=['dog', 'cat', 'owl']` all 0s are
                        replaced by dog, 1s by cat.
        model_name: (str) Model name. Defaults to 'Clusterer'

    Returns:
        None: To see plots, go to your W&B run page then expand the 'media' tab
              under 'auto visualizations'.

    Example:
    ```python
    wandb.sklearn.plot_clusterer(kmeans, X, cluster_labels, labels, "KMeans")
    ```
    """
    wandb.termlog('\nPlotting %s.' % model_name)
    if isinstance(model, sklearn.cluster.KMeans):
        elbow_curve(model, X_train)
        wandb.termlog('Logged elbow curve.')
        silhouette(model, X_train, cluster_labels, labels=labels, kmeans=True)
    else:
        silhouette(model, X_train, cluster_labels, kmeans=False)
    wandb.termlog('Logged silhouette plot.')