import numpy as np
from sklearn.base import clone
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding
from sklearn.neighbors import (
from sklearn.pipeline import make_pipeline
from sklearn.utils._testing import assert_array_almost_equal
def test_spectral_clustering():
    n_neighbors = 5
    X, _ = make_blobs(random_state=0)
    est_chain = make_pipeline(KNeighborsTransformer(n_neighbors=n_neighbors, mode='connectivity'), SpectralClustering(n_neighbors=n_neighbors, affinity='precomputed', random_state=42))
    est_compact = SpectralClustering(n_neighbors=n_neighbors, affinity='nearest_neighbors', random_state=42)
    labels_compact = est_compact.fit_predict(X)
    labels_chain = est_chain.fit_predict(X)
    assert_array_almost_equal(labels_chain, labels_compact)