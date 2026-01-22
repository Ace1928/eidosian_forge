import numpy as np
from numpy.testing import assert_, assert_equal
from scipy.linalg import clarkson_woodruff_transform
from scipy.linalg._sketches import cwt_matrix
from scipy.sparse import issparse, rand
from scipy.sparse.linalg import norm
def test_seed_returns_identical_transform_matrix(self):
    for A in self.test_matrices:
        for seed in self.seeds:
            S1 = cwt_matrix(self.n_sketch_rows, self.n_rows, seed=seed).toarray()
            S2 = cwt_matrix(self.n_sketch_rows, self.n_rows, seed=seed).toarray()
            assert_equal(S1, S2)