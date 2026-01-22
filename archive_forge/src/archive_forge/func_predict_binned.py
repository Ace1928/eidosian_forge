import numpy as np
from ._predictor import (
from .common import PREDICTOR_RECORD_DTYPE, Y_DTYPE
def predict_binned(self, X, missing_values_bin_idx, n_threads):
    """Predict raw values for binned data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The input samples.
        missing_values_bin_idx : uint8
            Index of the bin that is used for missing values. This is the
            index of the last bin and is always equal to max_bins (as passed
            to the GBDT classes), or equivalently to n_bins - 1.
        n_threads : int
            Number of OpenMP threads to use.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The raw predicted values.
        """
    out = np.empty(X.shape[0], dtype=Y_DTYPE)
    _predict_from_binned_data(self.nodes, X, self.binned_left_cat_bitsets, missing_values_bin_idx, n_threads, out)
    return out