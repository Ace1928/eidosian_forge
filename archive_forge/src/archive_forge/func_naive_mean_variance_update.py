import numpy as np
import pytest
from scipy import linalg, sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.special import expit
from sklearn.datasets import make_low_rank_matrix, make_sparse_spd_matrix
from sklearn.utils import gen_batches
from sklearn.utils._arpack import _init_arpack_v0
from sklearn.utils._testing import (
from sklearn.utils.extmath import (
from sklearn.utils.fixes import (
def naive_mean_variance_update(x, last_mean, last_variance, last_sample_count):
    updated_sample_count = last_sample_count + 1
    samples_ratio = last_sample_count / float(updated_sample_count)
    updated_mean = x / updated_sample_count + last_mean * samples_ratio
    updated_variance = last_variance * samples_ratio + (x - last_mean) * (x - updated_mean) / updated_sample_count
    return (updated_mean, updated_variance, updated_sample_count)