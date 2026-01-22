import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
Transform data by adding two virtual features.

        Parameters
        ----------
        X: numpy ndarray, {n_samples, n_components}
            New data, where n_samples is the number of samples and n_components
            is the number of components.
        y: None
            Unused

        Returns
        -------
        X_transformed: array-like, shape (n_samples, n_features)
            The transformed feature set
        