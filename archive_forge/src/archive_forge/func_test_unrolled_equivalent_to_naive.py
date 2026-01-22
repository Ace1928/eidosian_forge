import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.ensemble._hist_gradient_boosting.histogram import (
@pytest.mark.parametrize('constant_hessian', [True, False])
def test_unrolled_equivalent_to_naive(constant_hessian):
    rng = np.random.RandomState(42)
    n_samples = 10
    n_bins = 5
    sample_indices = np.arange(n_samples).astype(np.uint32)
    binned_feature = rng.randint(0, n_bins - 1, size=n_samples, dtype=np.uint8)
    ordered_gradients = rng.randn(n_samples).astype(G_H_DTYPE)
    if constant_hessian:
        ordered_hessians = np.ones(n_samples, dtype=G_H_DTYPE)
    else:
        ordered_hessians = rng.lognormal(size=n_samples).astype(G_H_DTYPE)
    hist_gc_root = np.zeros((1, n_bins), dtype=HISTOGRAM_DTYPE)
    hist_ghc_root = np.zeros((1, n_bins), dtype=HISTOGRAM_DTYPE)
    hist_gc = np.zeros((1, n_bins), dtype=HISTOGRAM_DTYPE)
    hist_ghc = np.zeros((1, n_bins), dtype=HISTOGRAM_DTYPE)
    hist_naive = np.zeros((1, n_bins), dtype=HISTOGRAM_DTYPE)
    _build_histogram_root_no_hessian(0, binned_feature, ordered_gradients, hist_gc_root)
    _build_histogram_root(0, binned_feature, ordered_gradients, ordered_hessians, hist_ghc_root)
    _build_histogram_no_hessian(0, sample_indices, binned_feature, ordered_gradients, hist_gc)
    _build_histogram(0, sample_indices, binned_feature, ordered_gradients, ordered_hessians, hist_ghc)
    _build_histogram_naive(0, sample_indices, binned_feature, ordered_gradients, ordered_hessians, hist_naive)
    hist_naive = hist_naive[0]
    hist_gc_root = hist_gc_root[0]
    hist_ghc_root = hist_ghc_root[0]
    hist_gc = hist_gc[0]
    hist_ghc = hist_ghc[0]
    for hist in (hist_gc_root, hist_ghc_root, hist_gc, hist_ghc):
        assert_array_equal(hist['count'], hist_naive['count'])
        assert_allclose(hist['sum_gradients'], hist_naive['sum_gradients'])
    for hist in (hist_ghc_root, hist_ghc):
        assert_allclose(hist['sum_hessians'], hist_naive['sum_hessians'])
    for hist in (hist_gc_root, hist_gc):
        assert_array_equal(hist['sum_hessians'], np.zeros(n_bins))