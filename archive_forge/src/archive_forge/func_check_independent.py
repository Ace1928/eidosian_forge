from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def check_independent(basis):
    if len(basis) == 0:
        return
    try:
        import numpy as np
    except ImportError:
        return
    H = nx.Graph()
    for b in basis:
        nx.add_cycle(H, b)
    inc = nx.incidence_matrix(H, oriented=True)
    rank = np.linalg.matrix_rank(inc.toarray(), tol=None, hermitian=False)
    assert inc.shape[1] - rank == len(basis)