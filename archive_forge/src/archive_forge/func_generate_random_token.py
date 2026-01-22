import pytest
from pytest import raises as assert_raises
import numpy as np
from scipy.cluster.hierarchy import DisjointSet
import string
def generate_random_token():
    k = len(string.ascii_letters)
    tokens = list(np.arange(k, dtype=int))
    tokens += list(np.arange(k, dtype=float))
    tokens += list(string.ascii_letters)
    tokens += [None for i in range(k)]
    tokens = np.array(tokens, dtype=object)
    rng = np.random.RandomState(seed=0)
    while 1:
        size = rng.randint(1, 3)
        element = rng.choice(tokens, size)
        if size == 1:
            yield element[0]
        else:
            yield tuple(element)