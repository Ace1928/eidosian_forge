import itertools
import random
import pytest
import cirq
import cirq.contrib.acquaintance as cca
def random_index_pairs(n_pairs: int):
    indices = list(range(2 * n_pairs))
    random.shuffle(indices)
    return tuple((indices[2 * i:2 * (i + 1)] for i in range(n_pairs)))