import joblib
import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KDTree, KernelDensity, NearestNeighbors
from sklearn.neighbors._ball_tree import kernel_norm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import assert_allclose
def test_kde_sample_weights():
    n_samples = 400
    size_test = 20
    weights_neutral = np.full(n_samples, 3.0)
    for d in [1, 2, 10]:
        rng = np.random.RandomState(0)
        X = rng.rand(n_samples, d)
        weights = 1 + (10 * X.sum(axis=1)).astype(np.int8)
        X_repetitions = np.repeat(X, weights, axis=0)
        n_samples_test = size_test // d
        test_points = rng.rand(n_samples_test, d)
        for algorithm in ['auto', 'ball_tree', 'kd_tree']:
            for metric in ['euclidean', 'minkowski', 'manhattan', 'chebyshev']:
                if algorithm != 'kd_tree' or metric in KDTree.valid_metrics:
                    kde = KernelDensity(algorithm=algorithm, metric=metric)
                    kde.fit(X, sample_weight=weights_neutral)
                    scores_const_weight = kde.score_samples(test_points)
                    sample_const_weight = kde.sample(random_state=1234)
                    kde.fit(X)
                    scores_no_weight = kde.score_samples(test_points)
                    sample_no_weight = kde.sample(random_state=1234)
                    assert_allclose(scores_const_weight, scores_no_weight)
                    assert_allclose(sample_const_weight, sample_no_weight)
                    kde.fit(X, sample_weight=weights)
                    scores_weight = kde.score_samples(test_points)
                    sample_weight = kde.sample(random_state=1234)
                    kde.fit(X_repetitions)
                    scores_ref_sampling = kde.score_samples(test_points)
                    sample_ref_sampling = kde.sample(random_state=1234)
                    assert_allclose(scores_weight, scores_ref_sampling)
                    assert_allclose(sample_weight, sample_ref_sampling)
                    diff = np.max(np.abs(scores_no_weight - scores_weight))
                    assert diff > 0.001
                    scale_factor = rng.rand()
                    kde.fit(X, sample_weight=scale_factor * weights)
                    scores_scaled_weight = kde.score_samples(test_points)
                    assert_allclose(scores_scaled_weight, scores_weight)