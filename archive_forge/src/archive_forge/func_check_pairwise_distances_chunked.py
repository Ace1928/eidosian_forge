import warnings
from types import GeneratorType
import numpy as np
from numpy import linalg
from scipy.sparse import issparse
from scipy.spatial.distance import (
import pytest
from sklearn import config_context
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics.pairwise import (
from sklearn.preprocessing import normalize
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
from sklearn.utils.parallel import Parallel, delayed
def check_pairwise_distances_chunked(X, Y, working_memory, metric='euclidean'):
    gen = pairwise_distances_chunked(X, Y, working_memory=working_memory, metric=metric)
    assert isinstance(gen, GeneratorType)
    blockwise_distances = list(gen)
    Y = X if Y is None else Y
    min_block_mib = len(Y) * 8 * 2 ** (-20)
    for block in blockwise_distances:
        memory_used = block.nbytes
        assert memory_used <= max(working_memory, min_block_mib) * 2 ** 20
    blockwise_distances = np.vstack(blockwise_distances)
    S = pairwise_distances(X, Y, metric=metric)
    assert_allclose(blockwise_distances, S, atol=1e-07)