import numpy as np
from ...base import BaseEstimator, TransformerMixin
from ...utils import check_array, check_random_state
from ...utils._openmp_helpers import _openmp_effective_n_threads
from ...utils.fixes import percentile
from ...utils.validation import check_is_fitted
from ._binning import _map_to_bins
from ._bitset import set_bitset_memoryview
from .common import ALMOST_INF, X_BINNED_DTYPE, X_BITSET_INNER_DTYPE, X_DTYPE
def make_known_categories_bitsets(self):
    """Create bitsets of known categories.

        Returns
        -------
        - known_cat_bitsets : ndarray of shape (n_categorical_features, 8)
            Array of bitsets of known categories, for each categorical feature.
        - f_idx_map : ndarray of shape (n_features,)
            Map from original feature index to the corresponding index in the
            known_cat_bitsets array.
        """
    categorical_features_indices = np.flatnonzero(self.is_categorical_)
    n_features = self.is_categorical_.size
    n_categorical_features = categorical_features_indices.size
    f_idx_map = np.zeros(n_features, dtype=np.uint32)
    f_idx_map[categorical_features_indices] = np.arange(n_categorical_features, dtype=np.uint32)
    known_categories = self.bin_thresholds_
    known_cat_bitsets = np.zeros((n_categorical_features, 8), dtype=X_BITSET_INNER_DTYPE)
    for mapped_f_idx, f_idx in enumerate(categorical_features_indices):
        for raw_cat_val in known_categories[f_idx]:
            set_bitset_memoryview(known_cat_bitsets[mapped_f_idx], raw_cat_val)
    return (known_cat_bitsets, f_idx_map)