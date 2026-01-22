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
def test_kde_pipeline_gridsearch():
    X, _ = make_blobs(cluster_std=0.1, random_state=1, centers=[[0, 1], [1, 0], [0, 0]])
    pipe1 = make_pipeline(StandardScaler(with_mean=False, with_std=False), KernelDensity(kernel='gaussian'))
    params = dict(kerneldensity__bandwidth=[0.001, 0.01, 0.1, 1, 10])
    search = GridSearchCV(pipe1, param_grid=params)
    search.fit(X)
    assert search.best_params_['kerneldensity__bandwidth'] == 0.1